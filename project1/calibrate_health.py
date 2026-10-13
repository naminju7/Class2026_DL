"""health logit-bias calibration for the train-only R3 model.

독립 valid에서 health logit 경계만 조정해 Macro-F1 변화를 측정한다.
- crop 예측/가중치는 건드리지 않는다.
- b를 health class 1(Diseased) logit에 더한다: health_logits[:, 1] += b
- 6개 (crop, health) 조합을 유지하는 반복 split-half 검증으로 과적합 위험도 함께 출력한다.
- 선택적으로 같은 구조의 다른 체크포인트에 선택된 bias를 적용해 새 state_dict를 저장한다.

예:
  python -u calibrate_health.py --weights ckpt/R3.pt
  python -u calibrate_health.py --weights ckpt/R3.pt \
      --apply_weights ckpt/R3_full27_ema.pt --out ckpt/R3_full27_ema_cal.pt
"""
import argparse
import importlib
import os

import torch
from torch.utils.data import DataLoader

from dataset import PlantDataset, get_transforms
from train import confusion, macro_f1


@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    crop_true, health_true, health_logits = [], [], []
    for images, crops, healths in loader:
        images = images.to(device)
        _, h = model(images)
        crop_true.append(crops.cpu())
        health_true.append(healths.cpu())
        health_logits.append(h.float().cpu())
    return torch.cat(crop_true), torch.cat(health_true), torch.cat(health_logits)


def health_f1(y, logits, bias, idx=None):
    if idx is not None:
        y = y[idx]
        logits = logits[idx]
    pred = (logits[:, 1] + float(bias) > logits[:, 0]).long()
    return macro_f1(confusion(y, pred, 2))[0]


def best_bias(y, logits, grid, idx=None):
    best_b, best_f = 0.0, -1.0
    eps = 1e-12
    for b in grid:
        b = float(b)
        f = health_f1(y, logits, b, idx)
        if f > best_f + eps or (abs(f - best_f) <= eps and abs(b) < abs(best_b)):
            best_b, best_f = b, f
    return best_b, best_f


def stratified_half(crop, health, generator):
    a, b = [], []
    strata = crop * 2 + health
    for s in range(6):
        idx = torch.where(strata == s)[0]
        if len(idx) == 0:
            continue
        idx = idx[torch.randperm(len(idx), generator=generator)]
        cut = len(idx) // 2
        a.append(idx[:cut])
        b.append(idx[cut:])
    return torch.cat(a), torch.cat(b)


def quantile(values, q):
    x = torch.tensor(values, dtype=torch.float32)
    return float(torch.quantile(x, torch.tensor(float(q))))


def apply_bias(weights_path, out_path, bias):
    state = torch.load(weights_path, map_location="cpu")
    key = "health_head.bias"
    if key not in state or state[key].numel() != 2:
        raise SystemExit(
            f"{weights_path}: '{key}' (길이 2)를 찾지 못했습니다. "
            "model_v1_baseline 계열 체크포인트만 자동 보정할 수 있습니다."
        )
    state[key] = state[key].clone()
    state[key][1] += float(bias)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(state, out_path)
    print(f"\n저장: {out_path}")
    print(f"  {key}[1] += {bias:+.4f}")
    print("  모델 구조/파라미터 수/state_dict 키는 그대로입니다.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="student_data_128")
    ap.add_argument("--model_module", default="model_v1_baseline")
    ap.add_argument("--weights", required=True,
                    help="bias를 고를 독립 valid용 체크포인트 (권장: train-only R3.pt)")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--bmin", type=float, default=-0.60)
    ap.add_argument("--bmax", type=float, default=0.60)
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--apply_weights", default=None,
                    help="선택된 bias를 적용할 같은 구조의 체크포인트 (예: R3_full27_ema.pt)")
    ap.add_argument("--out", default=None, help="보정 state_dict 저장 경로")
    args = ap.parse_args()

    if (args.apply_weights is None) != (args.out is None):
        raise SystemExit("--apply_weights 와 --out 은 함께 지정하세요.")
    if args.step <= 0 or args.bmax < args.bmin:
        raise SystemExit("bias grid 범위를 확인하세요.")

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    ds = PlantDataset(args.data_dir, "valid", get_transforms(train=False))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model_cls = importlib.import_module(args.model_module).MultiTaskCNN
    model = model_cls().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device), strict=True)

    crop, health, logits = collect_logits(model, loader, device)
    grid = torch.arange(args.bmin, args.bmax + args.step * 0.5, args.step)

    base = health_f1(health, logits, 0.0)
    full_b, full_f = best_bias(health, logits, grid)

    print(f"weights: {args.weights}")
    print(f"valid n={len(health)} | health Macro-F1 @ b=0: {base:.6f}")
    print(f"full-valid best: b={full_b:+.4f} | health F1 {full_f:.6f} | delta {full_f-base:+.6f}")
    print("정의: class 1(Diseased) logit에 b를 더함. b<0이면 Healthy 쪽으로 경계를 이동.")

    gen = torch.Generator().manual_seed(args.seed)
    deltas, chosen = [], []
    for _ in range(args.repeats):
        fit_idx, hold_idx = stratified_half(crop, health, gen)
        b, _ = best_bias(health, logits, grid, fit_idx)
        f0 = health_f1(health, logits, 0.0, hold_idx)
        f1 = health_f1(health, logits, b, hold_idx)
        chosen.append(b)
        deltas.append(f1 - f0)

    d = torch.tensor(deltas)
    pos = float((d > 0).float().mean())
    nonneg = float((d >= 0).float().mean())
    mean = float(d.mean())
    std = float(d.std(unbiased=False))
    med_b = quantile(chosen, 0.5)
    q25_b, q75_b = quantile(chosen, 0.25), quantile(chosen, 0.75)

    print(f"\n반복 split-half ({args.repeats}회, 6개 crop×health 조합 층화)")
    print(f"holdout delta: mean {mean:+.6f} ± {std:.6f}")
    print(f"P(delta>0)={pos:.2f}, P(delta>=0)={nonneg:.2f}")
    print(f"선택 bias: median {med_b:+.4f}, IQR [{q25_b:+.4f}, {q75_b:+.4f}]")

    robust = (mean >= 0.0015 and pos >= 0.70 and full_f - base >= 0.0020)
    if robust:
        print("\n판정: PASS — 독립 절반에서도 이득이 반복됩니다. 작은 bias 전이를 검토할 근거가 있습니다.")
    else:
        print("\n판정: REJECT/WEAK — valid에서 보이는 이득이 작거나 split-half에서 안정적이지 않습니다.")
        print("Public에 bias 후보를 바로 제출하기보다 R3/full EMA를 유지하는 쪽이 안전합니다.")

    if args.apply_weights is not None:
        print("\n주의: 아래 전이는 R3 valid에서 고른 bias를 다른 체크포인트에 적용하는 것이므로")
        print("logit scale이 비슷하다는 가정이 필요합니다. PASS일 때만 제출 후보로 쓰는 것을 권장합니다.")
        apply_bias(args.apply_weights, args.out, full_b)


if __name__ == "__main__":
    main()
