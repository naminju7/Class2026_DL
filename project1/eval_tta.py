import argparse
import time

import torch
from torch.utils.data import DataLoader

from dataset import PlantDataset, get_transforms
from model_v1_baseline import MultiTaskCNN


def confusion(true, pred, k):
    return torch.bincount(true * k + pred, minlength=k * k).view(k, k)


def macro_f1(conf):
    tp = conf.diag().float()
    precision = tp / conf.sum(0).clamp(min=1)
    recall = tp / conf.sum(1).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
    return f1.mean().item(), f1.tolist()


@torch.no_grad()
def evaluate_tta(model, loader, device, flips):
    model.eval()
    crop_true, crop_pred, health_true, health_pred = [], [], [], []
    start = time.time()

    for images, crops, healths in loader:
        images = images.to(device, non_blocking=True)
        crop_sum = None
        health_sum = None

        for dims in flips:
            x = images if dims is None else images.flip(dims)
            c, h = model(x)
            crop_sum = c if crop_sum is None else crop_sum + c
            health_sum = h if health_sum is None else health_sum + h

        crop_out = crop_sum / len(flips)
        health_out = health_sum / len(flips)

        crop_true.append(crops)
        crop_pred.append(crop_out.argmax(1).cpu())
        health_true.append(healths)
        health_pred.append(health_out.argmax(1).cpu())

    ct = torch.cat(crop_true)
    cp = torch.cat(crop_pred)
    ht = torch.cat(health_true)
    hp = torch.cat(health_pred)

    crop_conf = confusion(ct, cp, 3)
    health_conf = confusion(ht, hp, 2)
    crop_f1, crop_each = macro_f1(crop_conf)
    health_f1, health_each = macro_f1(health_conf)

    return {
        "crop_f1": crop_f1,
        "health_f1": health_f1,
        "score": (crop_f1 + health_f1) / 2,
        "crop_each": crop_each,
        "health_each": health_each,
        "crop_conf": crop_conf.tolist(),
        "health_conf": health_conf.tolist(),
        "seconds": time.time() - start,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="ckpt/A.pt")
    parser.add_argument("--data_dir", default="student_data_128")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"device: {device}")

    ds = PlantDataset(args.data_dir, "valid", get_transforms(train=False))
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    model = MultiTaskCNN().to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state, strict=True)

    modes = {
        "none": [None],
        "hflip": [None, (3,)],
        "vflip": [None, (2,)],
        "flip4": [None, (3,), (2,), (2, 3)],
    }

    results = {}
    print(f"valid: {len(ds)}장 | weights: {args.weights}")
    for name, flips in modes.items():
        r = evaluate_tta(model, loader, device, flips)
        results[name] = r
        print(
            f"{name:6s} | crop {r['crop_f1']:.4f} | health {r['health_f1']:.4f} "
            f"| avg {r['score']:.4f} | {r['seconds']:.1f}s"
        )
        print(f"         crop class F1: {[round(x, 4) for x in r['crop_each']]}")
        print(f"         health class F1: {[round(x, 4) for x in r['health_each']]}")

    best = max(results, key=lambda k: results[k]["score"])
    base = results["none"]["score"]
    print("\n===== TTA 비교 결과 =====")
    print(f"기준(none): {base:.4f}")
    print(f"최고 모드 : {best} = {results[best]['score']:.4f}")
    print(f"변화      : {results[best]['score'] - base:+.4f}")
    print("※ valid 개선이 아주 작으면 Public/Private에서도 반드시 좋아진다는 보장은 없습니다.")


if __name__ == "__main__":
    main()
