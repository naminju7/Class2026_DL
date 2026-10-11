"""재학습 없이 학습된 모델(A.pt)에 flip TTA 만 붙여서 valid 전체로 평가한다.

사용법 (project1 폴더에서)
  python -u eval_tta.py --weights ckpt/A.pt --export_dir submit_A_tta

- model_v1_tta.py 는 model_v1_baseline.py 와 state_dict 가 같아서 A.pt 를 그대로(strict) 불러온다.
- 네 가지 모드를 한 번에 비교한다:  none(원본) / h(원본+좌우) / v(원본+상하) / hv(원본+좌우+상하+좌우상하)
  logits 평균을 쓰고, 90도 회전은 학습에 쓰지 않았으므로 제외한다.
- valid 2284장 전체에서 crop/health/평균 Macro-F1 을 비교하고, 같은 표본을 다시 뽑는 부트스트랩으로
  "원본 대비 차이가 노이즈 범위 밖인가" 도 같이 보여 준다.
- CPU(채점 서버와 같은 조건)에서 걸리는 시간을 측정한다.
- 원본보다 평균 F1 이 좋아진 모드가 있으면 --export_dir 에 제출용 model.py + best_model.pt 를 만든다.
"""
import argparse
import copy
import os
import shutil
import time

import torch
from torch.utils.data import DataLoader

import model_v1_tta as M
from dataset import PlantDataset, get_transforms, load_class_names
from train import confusion, macro_f1

VIEW_DIMS = {"o": None, "h": (3,), "v": (2,), "hv": (2, 3)}     # 입력을 뒤집는 차원
MODE_VIEWS = {"none": ["o"], "h": ["o", "h"], "v": ["o", "v"], "hv": ["o", "h", "v", "hv"]}
REPORTED = {"crop_f1": 0.9789, "health_f1": 0.9908, "avg": 0.9848}  # 사용자가 보고한 A 의 valid 점수 (재현 확인용)


def avg_f1(ct, cp, ht, hp, idx, num_crops, num_health):
    c = macro_f1(confusion(ct[idx], cp[idx], num_crops))[0]
    h = macro_f1(confusion(ht[idx], hp[idx], num_health))[0]
    return (c + h) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="ckpt/A.pt")
    ap.add_argument("--data_dir", default="student_data_128")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--bootstrap", type=int, default=1000, help="부트스트랩 반복 수 (0 이면 생략)")
    ap.add_argument("--time_n", type=int, default=256, help="CPU 속도 측정에 쓸 이미지 수 (0 이면 생략)")
    ap.add_argument("--export_dir", default=None, help="TTA 가 더 좋을 때 제출용 파일을 만들 폴더")
    ap.add_argument("--export_mode", default="auto", choices=["auto", "none", "h", "v", "hv"],
                    help="auto: valid 평균 F1 이 가장 좋은 모드 (원본보다 좋아질 때만 export)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    crop_names, health_names = load_class_names(args.data_dir)
    nc, nh = len(crop_names), len(health_names)

    # ---------- 1. A.pt 를 TTA 모델 구조로 strict load (키가 하나라도 다르면 여기서 에러) ----------
    model = M.MultiTaskCNN(num_crops=nc, num_health=nh, tta_mode="none")
    try:
        state = torch.load(args.weights, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state)
    n_params = sum(p.numel() for p in model.parameters())
    n_buf = sum(b.numel() for b in model.buffers() if b.is_floating_point())
    print(f"'{args.weights}' load 성공 (strict). 파라미터 {n_params:,} + 버퍼 {n_buf:,} = {n_params + n_buf:,}, device={device}")
    model.to(device).eval()

    # ---------- 2. 네 가지 뷰의 logits 를 한 번만 계산해서 보관 ----------
    ds = PlantDataset(args.data_dir, args.split, get_transforms(train=False))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    clog = {k: [] for k in VIEW_DIMS}
    hlog = {k: [] for k in VIEW_DIMS}
    ct, ht = [], []
    first_x = None
    with torch.no_grad():
        for images, crops, healths in loader:
            x = images.to(device)
            if first_x is None:
                first_x = x.clone()
            for k, dims in VIEW_DIMS.items():
                c, h = model._forward(x if dims is None else x.flip(dims))
                clog[k].append(c.cpu())
                hlog[k].append(h.cpu())
            ct.append(crops)
            ht.append(healths)
    clog = {k: torch.cat(v) for k, v in clog.items()}
    hlog = {k: torch.cat(v) for k, v in hlog.items()}
    ct, ht = torch.cat(ct), torch.cat(ht)
    N = len(ct)
    print(f"{args.split}: {N}장 평가")

    # ---------- 3. 모드별 점수 (logits 평균) ----------
    preds, res = {}, {}
    for mode, views in MODE_VIEWS.items():
        cl = torch.stack([clog[v] for v in views]).mean(0)
        hl = torch.stack([hlog[v] for v in views]).mean(0)
        cp, hp = cl.argmax(1), hl.argmax(1)
        preds[mode] = (cp, hp)
        cf, hf = macro_f1(confusion(ct, cp, nc)), macro_f1(confusion(ht, hp, nh))
        res[mode] = {"crop": cf[0], "crop_pc": cf[1], "health": hf[0], "health_pc": hf[1], "avg": (cf[0] + hf[0]) / 2,
                     "crop_acc": (cp == ct).float().mean().item(), "health_acc": (hp == ht).float().mean().item()}

    # ---------- 4. 모델 클래스의 forward 가 위 계산과 같은지 확인 ----------
    with torch.no_grad():
        for mode, views in MODE_VIEWS.items():
            model.tta_mode = mode
            c, h = model(first_x)
            n = len(first_x)
            ref_c = torch.stack([clog[v][:n] for v in views]).mean(0)
            ref_h = torch.stack([hlog[v][:n] for v in views]).mean(0)
            err = max((c.cpu() - ref_c).abs().max().item(), (h.cpu() - ref_h).abs().max().item())
            assert err < 1e-3, f"모드 {mode}: forward 와 캐시 평균이 다름 (max err {err})"
            assert tuple(c.shape) == (n, nc) and tuple(h.shape) == (n, nh)
    print("model_v1_tta.forward 와 logits 평균 계산이 일치함 (출력 shape 도 (B,3), (B,2))")

    # ---------- 5. 부트스트랩 (같은 표본으로 원본과 쌍으로 비교) ----------
    boot = {}
    if args.bootstrap > 0:
        g = torch.Generator().manual_seed(0)
        sc = {m: [] for m in MODE_VIEWS}
        for _ in range(args.bootstrap):
            idx = torch.randint(0, N, (N,), generator=g)
            for m in MODE_VIEWS:
                sc[m].append(avg_f1(ct, preds[m][0], ht, preds[m][1], idx, nc, nh))
        sc = {m: torch.tensor(v) for m, v in sc.items()}
        for m in MODE_VIEWS:
            d = sc[m] - sc["none"]
            boot[m] = (d.mean().item(), d.quantile(0.025).item(), d.quantile(0.975).item(), (d > 0).float().mean().item(),
                       (d == 0).float().mean().item(), sc[m].std().item())

    # ---------- 6. 표 출력 ----------
    cn = " ".join(n[:1] for n in crop_names)
    hn = " ".join(n[:1] for n in health_names)
    print(f"\n{'모드':5s} {'뷰':>2s} | crop F1 ({cn}) | health F1 ({hn}) | 평균 F1 | 원본 대비 | 95% 구간 | P(개선)")
    base = res["none"]["avg"]
    for mode, views in MODE_VIEWS.items():
        r = res[mode]
        cpc = " ".join(f"{v:.3f}" for v in r["crop_pc"])
        hpc = " ".join(f"{v:.3f}" for v in r["health_pc"])
        line = (f"{mode:5s} {len(views):2d} | {r['crop']:.4f} ({cpc}) | {r['health']:.4f} ({hpc}) | "
                f"{r['avg']:.4f} | {r['avg'] - base:+.4f}")
        if mode in boot and mode != "none":
            mean, lo, hi, p, tie, _ = boot[mode]
            line += f" | [{lo:+.4f}, {hi:+.4f}] | {p:.2f} (동률 {tie:.2f})"
        print(line)
    if "none" in boot:
        print(f"(원본 점수의 부트스트랩 표준편차 ≈ {boot['none'][5]:.4f} -> 이보다 작은 차이는 노이즈로 보세요)")

    r0 = res["none"]
    ok = (abs(r0["crop"] - REPORTED["crop_f1"]) < 5e-4 and abs(r0["health"] - REPORTED["health_f1"]) < 5e-4)
    print(f"\n재현 확인(none): crop {r0['crop']:.4f}/{REPORTED['crop_f1']}, health {r0['health']:.4f}/{REPORTED['health_f1']}, "
          f"평균 {r0['avg']:.4f}/{REPORTED['avg']} -> {'일치' if ok else '차이가 있음(GPU/CPU 수치 차이 또는 다른 체크포인트인지 확인)'}")

    # ---------- 7. 채점 속도 (CPU, check_submission.py 와 같은 조건) ----------
    if args.time_n > 0:
        n = min(args.time_n, N)
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
        cpu_model = copy.deepcopy(model).cpu().eval()
        x = torch.stack([ds[i][0] for i in range(n)])
        print(f"\nCPU 속도 ({n}장 측정 -> {N}장 환산, 스레드 {torch.get_num_threads()}개)")
        with torch.no_grad():
            cpu_model.tta_mode = "none"
            cpu_model(x[:8])  # warm-up
            for mode in MODE_VIEWS:
                cpu_model.tta_mode = mode
                t0 = time.time()
                for i in range(0, n, 128):
                    cpu_model(x[i:i + 128])
                dt = time.time() - t0
                print(f"  {mode:5s} ({len(MODE_VIEWS[mode])}뷰): {dt:6.1f}s -> {N}장 약 {dt * N / n:6.0f}s")

    # ---------- 8. 후보 선택과 제출용 파일 ----------
    order = sorted(MODE_VIEWS, key=lambda m: (-res[m]["avg"], len(MODE_VIEWS[m])))  # 점수 높은 순, 동점이면 뷰 적은 쪽
    best = order[0] if args.export_mode == "auto" else args.export_mode
    print(f"\n후보: {best}  (평균 F1 {res[best]['avg']:.4f}, 원본 {base:.4f})")
    if best == "none" or res[best]["avg"] <= base:
        print("-> TTA 가 원본(A)보다 좋아지지 않았습니다. A.pt + model_v1_baseline.py 를 그대로 쓰는 것이 낫습니다.")
        if args.export_mode == "auto":
            return
    if args.export_dir:
        os.makedirs(args.export_dir, exist_ok=True)
        src = open("model_v1_tta.py", encoding="utf-8").read()
        assert 'DEFAULT_TTA = "hv"' in src
        with open(os.path.join(args.export_dir, "model.py"), "w", encoding="utf-8") as f:
            f.write(src.replace('DEFAULT_TTA = "hv"', f'DEFAULT_TTA = "{best}"'))
        shutil.copyfile(args.weights, os.path.join(args.export_dir, "best_model.pt"))
        print(f"제출용 파일 생성: {args.export_dir}/model.py (기본 tta_mode='{best}'), {args.export_dir}/best_model.pt")
        print("제출 전 점검:")
        print(f"  python check_submission.py --model {args.export_dir}/model.py --weights {args.export_dir}/best_model.pt")
        print("  (check_submission 은 TTA 가 켜진 상태로 채점하므로 위 CPU 속도와 점수를 같이 확인하세요)")


if __name__ == "__main__":
    main()