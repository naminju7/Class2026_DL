"""제출 폴더 만들기: 모델 구조 파일 + 가중치 + (선택) TTA 기본값 -> <out>/model.py, <out>/best_model.pt

저장소의 model.py 는 건드리지 않고, 후보마다 별도 폴더를 만든다. check_submission.py 는 수정하지 않고 그대로 호출한다.

사용법 (project1 폴더에서)
  # A 구조 + A.pt + 4뷰 TTA
  python make_submission.py --module model_v1_tta --weights ckpt/A.pt --tta hv --out submit_A_hv --check
  # A 구조 + EMA 가중치 (TTA 없음)
  python make_submission.py --module model_v1_baseline --weights ckpt/G3_ema.pt --out submit_G3_ema --check
  # v2 구조 + EMA 가중치 + 2뷰 TTA
  python make_submission.py --module model --weights ckpt/V2m_ema.pt --tta on --out submit_V2m_ema --check

--tta   model_v1_tta: none/h/v/hv (기본 none)   model(v2): on/off (기본 off)   model_v1_baseline: 해당 없음
--check 만든 폴더로 check_submission.py 를 실행하고 걸린 시간을 출력한다 (CPU, 채점 서버와 같은 방식)
"""
import argparse
import os
import shutil
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True, choices=["model_v1_baseline", "model_v1_tta", "model"])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tta", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data_dir", default="student_data_128")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    src = open(args.module + ".py", encoding="utf-8").read()
    if args.module == "model_v1_tta":
        tta = args.tta or "none"
        if tta not in ("none", "h", "v", "hv"):
            raise SystemExit("model_v1_tta 의 --tta 는 none/h/v/hv 중 하나")
        old = 'DEFAULT_TTA = "hv"'
        assert old in src
        src = src.replace(old, f'DEFAULT_TTA = "{tta}"')
    elif args.module == "model":
        on = (args.tta or "off") in ("on", "1", "true", "True")
        old_false = "num_health=2, tta=False):"
        old_true = "num_health=2, tta=True):"
        assert (old_false in src) ^ (old_true in src)
        old = old_false if old_false in src else old_true
        src = src.replace(old, f"num_health=2, tta={on}):", 1)
    elif args.tta not in (None, "none", "off"):
        raise SystemExit("model_v1_baseline 은 TTA 가 없습니다 (--tta 생략)")

    if os.path.abspath(args.out) == os.path.abspath("."):
        raise SystemExit("--out 은 현재 폴더가 아닌 새 폴더여야 합니다 (model.py 를 덮어쓰지 않기 위해)")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "model.py"), "w", encoding="utf-8") as f:
        f.write(src)
    shutil.copyfile(args.weights, os.path.join(args.out, "best_model.pt"))
    print(f"{args.out}/model.py ({args.module}, tta={args.tta}), {args.out}/best_model.pt ({os.path.getsize(args.weights) / 1024 / 1024:.2f}MB)")

    if args.check:
        cmd = [sys.executable, "check_submission.py", "--model", os.path.join(args.out, "model.py"),
               "--weights", os.path.join(args.out, "best_model.pt"), "--data_dir", args.data_dir]
        t0 = time.time()
        rc = subprocess.call(cmd)
        print(f"check_submission.py 소요 시간 {time.time() - t0:.0f}s (종료 코드 {rc})")
        sys.exit(rc)
    print(f"제출 전 점검: python check_submission.py --model {args.out}/model.py --weights {args.out}/best_model.pt")


if __name__ == "__main__":
    main()