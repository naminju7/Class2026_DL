import argparse
import copy
import importlib
import json
import math
import os
import random
import time
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import PlantDataset, get_transforms, load_class_names


# ============================================================
# 평가 지표
# ============================================================
def accuracy(conf):
    """혼동행렬 -> 정확도"""
    return (conf.diag().sum() / conf.sum()).item()


def macro_f1(conf):
    """혼동행렬(conf[정답, 예측]) -> (클래스별 F1 의 평균, 클래스별 F1 리스트)"""
    tp = conf.diag().float()
    precision = tp / conf.sum(0).clamp(min=1)
    recall = tp / conf.sum(1).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
    return f1.mean().item(), f1.tolist()


def confusion(true, pred, k):
    """정답/예측 텐서 -> k x k 혼동행렬 (행: 정답, 열: 예측)"""
    return torch.bincount(true * k + pred, minlength=k * k).view(k, k)


def set_tta(model, flag):
    """model.py 에 flip TTA 스위치(tta)가 있으면 켜고 끈다 (v1 모델에는 없으므로 무시)"""
    if hasattr(model, "tta"):
        model.tta = bool(flag)


# ============================================================
# 평가 (채점 서버와 같이 fp32 로 계산)
# ============================================================
@torch.no_grad()
def evaluate(model, loader, device, num_crops, num_health):
    """모델을 loader 의 데이터로 평가한다.

    반환: dict
      loss        : 평균 손실 (가중치/label smoothing 없는 일반 CrossEntropy, 두 과제의 합)
      crop_acc, crop_f1, crop_f1_per_class, crop_conf
      health_acc, health_f1, health_f1_per_class, health_conf
      score       : 두 과제 Macro-F1 의 평균 (최고 모델 선택 기준)
      crop_true/crop_pred/health_true/health_pred : 샘플별 정답/예측 (오류 분석용)
    """
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    total_loss, n = 0.0, 0
    ct, cp, ht, hp = [], [], [], []

    for images, crops, healths in loader:
        images, crops, healths = images.to(device), crops.to(device), healths.to(device)
        crop_out, health_out = model(images)
        total_loss += (ce(crop_out, crops) + ce(health_out, healths)).item()
        n += images.size(0)
        ct.append(crops.cpu())
        cp.append(crop_out.argmax(1).cpu())
        ht.append(healths.cpu())
        hp.append(health_out.argmax(1).cpu())

    ct, cp, ht, hp = torch.cat(ct), torch.cat(cp), torch.cat(ht), torch.cat(hp)
    crop_conf = confusion(ct, cp, num_crops)
    health_conf = confusion(ht, hp, num_health)
    crop_f1, crop_f1_per_class = macro_f1(crop_conf)
    health_f1, health_f1_per_class = macro_f1(health_conf)
    return {
        "loss": total_loss / n,
        "crop_acc": accuracy(crop_conf),
        "crop_f1": crop_f1,
        "crop_f1_per_class": crop_f1_per_class,
        "crop_conf": crop_conf,
        "health_acc": accuracy(health_conf),
        "health_f1": health_f1,
        "health_f1_per_class": health_f1_per_class,
        "health_conf": health_conf,
        "score": (crop_f1 + health_f1) / 2,
        "crop_true": ct, "crop_pred": cp, "health_true": ht, "health_pred": hp,
    }


def print_report(result, crop_names, health_names):
    """evaluate() 결과를 보기 좋게 출력한다."""
    for title, key, names in [("작물 종류", "crop", crop_names),
                              ("건강 상태", "health", health_names)]:
        print(f"\n[{title}] Accuracy {result[key + '_acc']:.3f}, Macro-F1 {result[key + '_f1']:.3f}")
        for name, v in zip(names, result[key + "_f1_per_class"]):
            print(f"  {name:10s} F1 {v:.3f}")
        print("  혼동행렬 (행: 정답, 열: 예측)")
        print("  " + str(result[key + "_conf"].tolist()))

    # 정답 (작물, 상태) 조합별로 어디서 틀리는지 -> Apple 이 왜 약한지 확인용
    ct, cp = result["crop_true"], result["crop_pred"]
    ht, hp = result["health_true"], result["health_pred"]
    print("\n[정답 조합별 오류 분석]  (개수 | 작물 정답률 (틀린 곳) | 건강 정답률)")
    for c, cname in enumerate(crop_names):
        for h, hname in enumerate(health_names):
            m = (ct == c) & (ht == h)
            n = int(m.sum())
            if n == 0:
                continue
            crop_acc = (cp[m] == c).float().mean().item()
            health_acc = (hp[m] == h).float().mean().item()
            wrong = Counter(cp[m][cp[m] != c].tolist())
            wrong_s = ", ".join(f"->{crop_names[k]} {v}" for k, v in wrong.most_common()) or "오류 없음"
            print(f"  {cname:7s}/{hname:8s} n={n:4d} | 작물 {crop_acc:.3f} ({wrong_s}) | 건강 {health_acc:.3f}")
    print(f"\n평균 Macro-F1 (score): {result['score']:.4f}")


# ============================================================
# 학습 도구: 클래스 가중치, 샘플러, 스케줄러, EMA
# ============================================================
def class_weights(targets, num_classes, power, device):
    """클래스 빈도의 역수^power 가중치 (power=0 이면 None = 가중치 없음).
    power=0.5 는 '역수의 제곱근' 으로, 완전한 역수(1.0)보다 부드럽게 소수 클래스를 키운다."""
    if power <= 0:
        return None
    cnt = torch.bincount(torch.tensor(targets), minlength=num_classes).float().clamp(min=1)
    w = (cnt.sum() / (num_classes * cnt)) ** power
    return (w / w.mean()).to(device)


def make_sampler(labels, power):
    """(작물, 상태) 6개 조합의 빈도^-power 를 샘플 확률로 쓰는 WeightedRandomSampler"""
    cell_count = Counter(labels)
    w = torch.tensor([cell_count[l] ** (-power) for l in labels], dtype=torch.double)
    return WeightedRandomSampler(w, num_samples=len(labels), replacement=True)


def make_optimizer(model, args):
    if args.optim == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr)
    # AdamW: BatchNorm 의 weight/bias 와 모든 bias(1차원 파라미터)에는 weight decay 를 적용하지 않는다
    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]
    return torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                              {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)


def make_scheduler(optimizer, args, steps_per_epoch):
    total = steps_per_epoch * args.epochs
    if args.sched == "none":
        return None
    if args.sched == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, total_steps=total, pct_start=0.2,
            anneal_strategy="cos", div_factor=20, final_div_factor=200)
    # cosine: 처음 warmup epoch 동안 선형으로 올린 뒤 코사인으로 lr*min_lr_ratio 까지 감소 (매 step 갱신)
    warm = max(1, int(steps_per_epoch * args.warmup))

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        p = (step - warm) / max(1, total - warm)
        return args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EMA:
    """가중치의 지수이동평균 (torch 만으로 구현). 평가/저장은 self.module 로 한다.
    BatchNorm 통계(buffer)도 같이 평균낸다. 저장되는 state_dict 의 키는 원래 모델과 같다."""

    def __init__(self, model, decay):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))  # 초반에는 짧게 평균
        for e, m in zip(self.module.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(m.detach(), alpha=1 - d)
            else:
                e.copy_(m)


# ============================================================
# 학습 (전체): 매 epoch 평가, 최고 모델 저장
# ============================================================
def train(model, train_loader, eval_loader, device, args, crop_crit, health_crit,
          num_crops, num_health, save_path):
    """반환: epoch 별 기록 리스트 (학습 곡선을 그릴 때 사용)"""
    use_amp = bool(args.amp) and device == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    optimizer = make_optimizer(model, args)
    scheduler = make_scheduler(optimizer, args, len(train_loader))
    ema = EMA(model, args.ema) if args.ema > 0 else None
    cat = ["Apple", "Grape", "Tomato"]

    history = []
    best_score, best_src = -1.0, ""
    for epoch in range(1, args.epochs + 1):
        start = time.time()

        # ----- 1 epoch 학습 -----
        model.train()
        total_loss, n = 0.0, 0
        for images, crops, healths in train_loader:
            images = images.to(device, non_blocking=True)
            crops = crops.to(device, non_blocking=True)
            healths = healths.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                dtype=torch.float16, enabled=use_amp):
                crop_out, health_out = model(images)
            loss = (args.crop_w * crop_crit(crop_out.float(), crops)
                    + args.health_w * health_crit(health_out.float(), healths))

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            if ema is not None:
                ema.update(model)

            total_loss += loss.item() * images.size(0)
            n += images.size(0)
        train_loss = total_loss / n
        lr_now = optimizer.param_groups[0]["lr"]

        # ----- 평가 (원본 모델, EMA 모델) -----
        r = evaluate(model, eval_loader, device, num_crops, num_health)
        r_ema = evaluate(ema.module, eval_loader, device, num_crops, num_health) if ema else None

        pc = " ".join(f"{cat[i][0]}{v:.2f}" for i, v in enumerate(r["crop_f1_per_class"])) \
            if num_crops == 3 else ""
        line = (f"[{epoch:2d}/{args.epochs}] lr {lr_now:.2e} | "
                f"train_loss {train_loss:.4f} | valid_loss {r['loss']:.4f} | "
                f"crop F1 {r['crop_f1']:.4f} ({pc}) | health F1 {r['health_f1']:.4f} | "
                f"avg F1 {r['score']:.4f}")
        if r_ema is not None:
            line += (f" || EMA crop {r_ema['crop_f1']:.4f} health {r_ema['health_f1']:.4f} "
                     f"avg {r_ema['score']:.4f}")
        print(line + f" | {time.time() - start:.0f}s", flush=True)

        rec = {"epoch": epoch, "lr": lr_now, "train_loss": train_loss, "valid_loss": r["loss"],
               "crop_f1": r["crop_f1"], "health_f1": r["health_f1"], "score": r["score"]}
        if r_ema is not None:
            rec.update({"ema_crop_f1": r_ema["crop_f1"], "ema_health_f1": r_ema["health_f1"],
                        "ema_score": r_ema["score"]})
        history.append(rec)

        # ----- 최고 모델 저장 (valid 평균 Macro-F1 기준; 원본/EMA 중 더 좋은 쪽) -----
        if not args.save_last:
            cands = [("raw", model, r["score"])]
            if r_ema is not None:
                cands.append(("ema", ema.module, r_ema["score"]))
            for src, m, sc in cands:
                if sc > best_score:
                    best_score, best_src = sc, src
                    torch.save(m.state_dict(), save_path)
                    print(f"  -> 최고 모델 저장 ({src}, 평균 Macro-F1 {sc:.4f})", flush=True)

    if args.save_last:
        final = ema.module if ema is not None else model
        torch.save(final.state_dict(), save_path)
        print(f"  -> 마지막 epoch 모델 저장 ({'ema' if ema is not None else 'raw'})")
    return history


# ============================================================
# 메인
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="student_data_128")
    parser.add_argument("--name", default="run", help="실험 이름 (저장 파일 이름에 사용)")
    parser.add_argument("--model_module", default="model",
                        help="모델이 들어 있는 파이썬 파일 이름 (기본 model, 이전 모델은 model_v1_baseline)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3, help="(최대) 학습률")
    parser.add_argument("--optim", default="adamw", choices=["adam", "adamw"])
    parser.add_argument("--wd", type=float, default=0.05, help="AdamW weight decay")
    parser.add_argument("--sched", default="cosine", choices=["none", "cosine", "onecycle"])
    parser.add_argument("--warmup", type=float, default=2.0, help="cosine 의 warmup epoch 수")
    parser.add_argument("--min_lr_ratio", type=float, default=0.01)
    parser.add_argument("--clip", type=float, default=0.0, help="gradient clipping (0 이면 끔)")
    parser.add_argument("--aug", default="strong", choices=["basic", "medium", "strong"])
    parser.add_argument("--cw_power", type=float, default=0.5,
                        help="클래스 가중치 = 빈도^-power (0: 끔, 0.5: 제곱근, 1: 완전 역수)")
    parser.add_argument("--sampler", default="none", choices=["none", "cell_sqrt"],
                        help="cell_sqrt: (작물,상태) 6개 조합 빈도^-0.5 로 뽑는 WeightedRandomSampler")
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--crop_w", type=float, default=1.5, help="작물 loss 가중치")
    parser.add_argument("--health_w", type=float, default=1.0, help="건강 상태 loss 가중치")
    parser.add_argument("--ema", type=float, default=0.998, help="EMA decay (0 이면 끔)")
    parser.add_argument("--tta", type=int, default=1, help="평가 시 flip TTA (모델이 지원할 때) 1/0")
    parser.add_argument("--amp", type=int, default=1, help="GPU 에서 mixed precision 학습 1/0")
    parser.add_argument("--include_valid", action="store_true",
                        help="train+valid 를 모두 학습에 사용 (최종 제출용. valid 점수는 더 이상 믿을 수 없음)")
    parser.add_argument("--save_last", action="store_true",
                        help="최고 epoch 대신 마지막 epoch 모델 저장 (--include_valid 일 때 자동)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="auto: GPU 가 있으면 GPU, 없으면 CPU")
    parser.add_argument("--save_path", default=None, help="기본값: ckpt/<name>.pt")
    parser.add_argument("--eval_only", action="store_true",
                        help="학습하지 않고 save_path 의 모델을 불러와 평가만 한다")
    parser.add_argument("--eval_split", default="valid",
                        help="평가할 데이터 (data_dir 안의 <split>.csv, <split>/images)")
    parser.add_argument("--smoke", action="store_true", help="코드 점검용: 아주 작은 데이터로 2 epoch")
    args = parser.parse_args()

    if args.include_valid:
        args.save_last = True
    if args.save_path is None:
        os.makedirs("ckpt", exist_ok=True)
        args.save_path = os.path.join("ckpt", f"{args.name}.pt")
    elif os.path.dirname(args.save_path):
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    limit = None
    if args.smoke:
        args.epochs, args.batch_size, args.num_workers, limit = 2, 32, 0, 256

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"device: {device}")
    print(f"설정: {vars(args)}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    # ---------------- 데이터 / 모델 ----------------
    crop_names, health_names = load_class_names(args.data_dir)
    num_crops, num_health = len(crop_names), len(health_names)
    pin = device == "cuda"
    persistent = args.num_workers > 0

    eval_set = PlantDataset(args.data_dir, args.eval_split, get_transforms(train=False), limit=limit)
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=pin,
                             persistent_workers=persistent)

    model_cls = importlib.import_module(args.model_module).MultiTaskCNN
    model = model_cls(num_crops=num_crops, num_health=num_health).to(device)
    set_tta(model, args.tta)
    n_params = sum(p.numel() for p in model.parameters())
    n_buffers = sum(b.numel() for b in model.buffers() if b.is_floating_point())
    print(f"파라미터 {n_params:,} + 버퍼 {n_buffers:,} = {n_params + n_buffers:,} (제한 2,000,000, 권장 1,800,000 이하)")

    # ---------------- 평가만 하기 ----------------
    if args.eval_only:
        model.load_state_dict(torch.load(args.save_path, map_location=device))
        print(f"'{args.save_path}' 모델로 {args.eval_split} ({len(eval_set)}장) 평가 (tta={args.tta})")
        print_report(evaluate(model, eval_loader, device, num_crops, num_health), crop_names, health_names)
        return

    # ---------------- 학습 데이터 ----------------
    splits = ["train", "valid"] if args.include_valid else "train"
    train_set = PlantDataset(args.data_dir, splits, get_transforms(train=True, aug=args.aug), limit=limit)
    labels = train_set.labels
    sampler = make_sampler(labels, 0.5) if args.sampler == "cell_sqrt" else None
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=sampler is None,
                              sampler=sampler, num_workers=args.num_workers, pin_memory=pin,
                              persistent_workers=persistent, drop_last=True)
    print(f"train: {len(train_set)}장, {args.eval_split}: {len(eval_set)}장")
    if args.include_valid:
        print("!! train+valid 로 학습합니다. 아래 valid 점수는 학습에 쓴 데이터라 과대평가된 값입니다. "
              "마지막 epoch 모델을 저장합니다.")

    # ---------------- 손실 함수 ----------------
    cw_power = 0.0 if sampler is not None else args.cw_power  # 샘플러와 클래스 가중치를 같이 쓰지 않는다
    crop_w = class_weights([c for c, _ in labels], num_crops, cw_power, device)
    health_w = class_weights([h for _, h in labels], num_health, cw_power, device)
    print(f"클래스 가중치 crop: {None if crop_w is None else [round(v, 3) for v in crop_w.tolist()]}, "
          f"health: {None if health_w is None else [round(v, 3) for v in health_w.tolist()]}")
    crop_crit = nn.CrossEntropyLoss(weight=crop_w, label_smoothing=args.label_smoothing)
    health_crit = nn.CrossEntropyLoss(weight=health_w, label_smoothing=args.label_smoothing)

    history = train(model, train_loader, eval_loader, device, args, crop_crit, health_crit,
                    num_crops, num_health, args.save_path)
    with open(os.path.splitext(args.save_path)[0] + "_log.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "history": history}, f, ensure_ascii=False, indent=1)

    # ---------------- 저장된 모델로 최종 평가 ----------------
    model.load_state_dict(torch.load(args.save_path, map_location=device))
    print(f"\n===== 최종 결과 ({args.eval_split}, tta={args.tta}) =====")
    final = evaluate(model, eval_loader, device, num_crops, num_health)
    print_report(final, crop_names, health_names)
    if hasattr(model, "tta"):
        set_tta(model, not args.tta)
        other = evaluate(model, eval_loader, device, num_crops, num_health)
        print(f"\n[참고] tta={int(not args.tta)} 로 평가하면: crop F1 {other['crop_f1']:.4f}, "
              f"health F1 {other['health_f1']:.4f}, 평균 {other['score']:.4f}")
    print(f"\n저장 위치: {args.save_path}")
    print("제출하려면: cp %s best_model.pt && python check_submission.py" % args.save_path)


if __name__ == "__main__":
    main()