"""여러 체크포인트를 같은 valid 로 비교한다 (재학습 없음). 제출 후보를 고를 때 사용한다.

사용법 (project1 폴더에서, 첫 번째 항목이 기준(ref)):
  python -u compare.py model_v1_baseline:ckpt/A.pt \\
                       model_v1_tta:ckpt/A.pt:hv \\
                       model_v1_baseline:ckpt/G3_ema.pt \\
                       model_v1_baseline:ckpt/G3_swa.pt \\
                       model:ckpt/V2m_ema.pt:on

항목 형식  <model 파일 이름(.py 제외)>:<가중치 경로>[:<TTA>]
  TTA  model_v1_tta 는 none/h/v/hv, model.py(v2) 는 on/off.  생략하면 TTA 없음.

출력: 평균/crop/health Macro-F1, 기준 대비 차이와 95% 구간(같은 표본을 다시 뽑는 paired bootstrap),
      "기준보다 좋았던 비율", 기준과 예측이 달라진 이미지 수, 파라미터/파일 크기, CPU 추론 시간.
valid 점수 차이가 부트스트랩 구간 안에 0 이 들어 있으면 hidden test 에서도 좋아진다고 믿기 어렵다.
"""
import argparse
import copy
import importlib
import os
import time

import torch

from dataset import PlantDataset, get_transforms, load_class_names
from train import confusion, macro_f1

ON = {"on", "1", "true", "True"}


def load_spec(spec, nc, nh):
    parts = spec.split(":")
    if len(parts) < 2:
        raise SystemExit(f"항목 형식이 잘못되었습니다: {spec}  (<model>:<weights>[:<tta>])")
    module, weights = parts[0], parts[1]
    tta = parts[2] if len(parts) > 2 else None
    model = importlib.import_module(module).MultiTaskCNN(num_crops=nc, num_health=nh)
    try:
        state = torch.load(weights, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(weights, map_location="cpu")
    model.load_state_dict(state)  # strict: 구조가 다르면 여기서 에러
    if hasattr(model, "tta_mode"):
        model.tta_mode = tta or "none"
    elif hasattr(model, "tta"):
        model.tta = tta in ON
    elif tta not in (None, "none", "off"):
        print(f"[경고] {module} 은 TTA 를 지원하지 않아 '{tta}' 를 무시합니다")
    n_params = sum(p.numel() for p in model.parameters())
    n_buf = sum(b.numel() for b in model.buffers() if b.is_floating_point())
    return model.eval(), n_params + n_buf, os.path.getsize(weights) / 1024 / 1024


def avg_f1(ct, cp, ht, hp, idx, nc, nh):
    return (macro_f1(confusion(ct[idx], cp[idx], nc))[0] + macro_f1(confusion(ht[idx], hp[idx], nh))[0]) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+", help="<model>:<weights>[:<tta>] (첫 번째가 기준)")
    ap.add_argument("--data_dir", default="student_data_128")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--time_n", type=int, default=128, help="CPU 추론 시간 측정에 쓸 이미지 수 (0 이면 생략)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    crop_names, health_names = load_class_names(args.data_dir)
    nc, nh = len(crop_names), len(health_names)

    ds = PlantDataset(args.data_dir, args.split, get_transforms(train=False))
    X = torch.stack([ds[i][0] for i in range(len(ds))])
    ct = torch.tensor([c for _, c, _ in ds.items])
    ht = torch.tensor([h for _, _, h in ds.items])
    N = len(ds)
    print(f"{args.split} {N}장, device={device}\n")

    preds, rows = [], []
    for spec in args.specs:
        model, n_total, mb = load_spec(spec, nc, nh)
        m = copy.deepcopy(model).to(device)
        cps, hps = [], []
        with torch.no_grad():
            for i in range(0, N, args.batch_size):
                c, h = m(X[i:i + args.batch_size].to(device))
                cps.append(c.argmax(1).cpu())
                hps.append(h.argmax(1).cpu())
        cp, hp = torch.cat(cps), torch.cat(hps)
        cf, hf = macro_f1(confusion(ct, cp, nc)), macro_f1(confusion(ht, hp, nh))
        sec = None
        if args.time_n > 0:
            torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
            cm = copy.deepcopy(model).cpu().eval()
            n = min(args.time_n, N)
            with torch.no_grad():
                cm(X[:8])
                t0 = time.time()
                for i in range(0, n, 128):
                    cm(X[i:min(i + 128, n)])
            sec = (time.time() - t0) * N / n
        preds.append((cp, hp))
        rows.append({"spec": spec, "crop": cf[0], "crop_pc": cf[1], "health": hf[0], "health_pc": hf[1],
                     "avg": (cf[0] + hf[0]) / 2, "crop_err": int((cp != ct).sum()), "health_err": int((hp != ht).sum()),
                     "n_total": n_total, "mb": mb, "sec": sec})
        if n_total > 2_000_000 or mb > 10:
            print(f"[경고] {spec}: 크기 제한 초과 (파라미터+버퍼 {n_total:,}, 파일 {mb:.2f}MB)")

    # ---- paired bootstrap: 같은 표본 index 로 모든 후보를 평가해서 기준과의 차이를 본다 ----
    boot = None
    if args.bootstrap > 0:
        g = torch.Generator().manual_seed(0)
        sc = [[] for _ in rows]
        for _ in range(args.bootstrap):
            idx = torch.randint(0, N, (N,), generator=g)
            for k, (cp, hp) in enumerate(preds):
                sc[k].append(avg_f1(ct, cp, ht, hp, idx, nc, nh))
        sc = [torch.tensor(v) for v in sc]
        boot = [(sc[k] - sc[0], sc[k].std().item()) for k in range(len(rows))]

    ref = rows[0]
    print(f"{'#':>2s} {'평균 F1':>8s} {'crop':>7s} {'health':>7s} | {'기준 대비':>8s} {'95% 구간':>19s} {'P(개선)':>7s} | "
          f"{'예측 달라진 장수(crop/health)':>29s} | {'오류 장수(c/h)':>13s} | {'파라미터':>9s} {'MB':>5s} {'CPU초':>6s}")
    for k, r in enumerate(rows):
        diff_c = int((preds[k][0] != preds[0][0]).sum())
        diff_h = int((preds[k][1] != preds[0][1]).sum())
        if boot is not None and k > 0:
            d = boot[k][0]
            ci = f"[{d.quantile(0.025).item():+.4f},{d.quantile(0.975).item():+.4f}]"
            pb = f"{(d > 0).float().mean().item():.2f}"
        else:
            ci, pb = "-", "-"
        sec = f"{r['sec']:.0f}" if r["sec"] is not None else "-"
        print(f"{k:2d} {r['avg']:8.4f} {r['crop']:7.4f} {r['health']:7.4f} | {r['avg'] - ref['avg']:+8.4f} {ci:>19s} {pb:>7s} | "
              f"{diff_c:>13d}/{diff_h:<15d} | {r['crop_err']:>6d}/{r['health_err']:<6d} | {r['n_total']:9,d} {r['mb']:5.2f} {sec:>6s}")
    print()
    for k, r in enumerate(rows):
        cpc = " ".join(f"{n[:1]}{v:.3f}" for n, v in zip(crop_names, r["crop_pc"]))
        hpc = " ".join(f"{n[:1]}{v:.3f}" for n, v in zip(health_names, r["health_pc"]))
        print(f"[{k}] {r['spec']}\n     crop {cpc} | health {hpc}")
    if boot is not None:
        print(f"\n기준 점수의 부트스트랩 표준편차 ≈ {boot[0][1]:.4f}. 95% 구간에 0 이 들어 있거나 P(개선)이 0.9 미만이면 "
              f"valid 에서 좋아 보여도 hidden test 개선을 기대하기 어렵습니다 (예측이 달라진 장수가 적으면 더더욱).")


if __name__ == "__main__":
    main()