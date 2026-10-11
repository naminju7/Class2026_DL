# Public 0.9740 → 0.98+ : hidden-test 일반화 실험 계획

기준 A: `model_v1_baseline.py` + `ckpt/A.pt` (valid 평균 0.9848 = crop 0.9789 / health 0.9908, Public 0.9740)

> 이 문서의 숫자 중 A 의 valid/Public 외에는 아직 측정 전입니다. 각 단계의 결과는 Colab 에서 아래 명령어로 확인하세요.
> (코드를 작성한 환경에는 PyTorch 와 `A.pt` 가 없어서 학습/채점은 실행하지 못했고, 구조·파라미터 수·shape 는 가짜 torch 로 확인했습니다.)

## 0. 먼저: A 를 지킨다

```
!mkdir -p backup_A && cp ckpt/A.pt ckpt/A_log.json model_v1_baseline.py backup_A/
!chmod a-w ckpt/A.pt ckpt/A_log.json
!cp -r backup_A /content/drive/MyDrive/      # Drive 에도 한 부 (Colab 세션이 끝나면 ckpt 가 사라짐)
```
- 새 `train.py` 는 이미 있는 `ckpt/<name>.pt`, `_last/_ema/_swa.pt`, `_log.json` 을 **절대 덮어쓰지 않고** 종료합니다 (`--overwrite` 를 줘야만 덮어씀). `--smoke` 는 `ckpt/_smoke/` 에만 저장합니다.
- `train.py` 의 기본 옵션이 곧 A 설정입니다 (model_v1_baseline, 25 epoch, AdamW 2e-3, wd 0.05, cosine+warmup 2, basic 증강, class weight/label smoothing/EMA 없음, seed 42). 바뀌는 옵션만 적습니다.

## 1. 격차 원인 점검 (이미 해 본 것과 해야 할 것)

- **train↔valid 중복 누수는 원인이 아닙니다.** 32x32 정규화 상관(8방향 회전/뒤집기 포함)으로 보면 valid 이미지의 최근접 train 유사도 > 0.95 는 11장(0.5%)이고, train 안에서 자기 자신을 뺀 최근접 유사도의 비율(0.4%)과 같습니다. valid 가 특별히 부풀려진 것은 아닙니다.
- 남는 후보는 (1) 25개 epoch 중 최고점을 같은 valid 로 고른 **선택 편향**, (2) flip 만 쓴 **약한 증강**으로 인한 hidden test 의 작은 분포 차이 (3) 작은 클래스(Grape Healthy 53장 등)의 **표본 노이즈**입니다.
- A 의 선택 편향은 로그에서 바로 계산됩니다 (새 `train.py` 는 이 값을 자동 출력):
```
!python - <<'EOF'
import json, statistics as st
h = json.load(open("ckpt/A_log.json"))["history"]; s = [x["score"] for x in h]
print("best", max(s), "last5 mean", st.mean(s[-5:]), "std", st.pstdev(s[-5:]), "gap", max(s) - st.mean(s[-5:]))
EOF
```
  gap 이 0.003 이상이면 best epoch 점수는 낙관적이므로, Public 에서는 EMA/SWA 처럼 한 epoch 에 의존하지 않는 후보가 유리합니다.
- Public 점수 차이 해석 기준: `eval_tta.py`/`compare.py` 가 출력하는 부트스트랩 표준편차(valid 2284장 기준 대략 0.003~0.005 예상)보다 작은 차이는 노이즈입니다. **Public 이 0.003 오르내리는 것으로 설정을 바꾸지 마세요.**

## 2. 실험 순서

| 단계 | 재학습 | 명령어 | 무엇을 검증하나 |
|---|---|---|---|
| T | 없음 | `!python -u eval_tta.py --weights ckpt/A.pt --export_dir submit_A_tta` | A.pt + flip TTA (none / 원본+좌우 / 원본+상하 / 4뷰), CPU 채점 시간, 부트스트랩 |
| G2 | 있음 | `!python -u train.py --name G2 --aug mild --ema 0.998` | A + mild 증강(= G1) + EMA(= G2) + SWA. 한 번의 run 에서 **best / last / ema / swa 4개 후보** |
| G3 | 있음 | `!python -u train.py --name G3 --aug mild --ema 0.998 --epochs 32` | G2 와 같은데 32 epoch (증강이 들어가면 25 epoch 가 부족할 수 있음) |
| V2m | 있음 | `!python -u train.py --name V2m --model_module model --aug mild --ema 0.998 --epochs 32` | v2 구조 (A 와 같은 optimizer/scheduler, strong 증강/RandomErasing/class weight 없음) |
| 비교 | 없음 | 아래 `compare.py` | A 대비 차이가 노이즈 밖인지 |

- EMA 는 학습 경로를 바꾸지 않으므로 **G1(=증강만) 은 G2 run 의 `best`/`last` 후보와 같습니다.** G1 을 따로 돌릴 필요가 없습니다.
- `mild` 증강: 좌우/상하 뒤집기 + 90도 회전(무손실) + 50% 확률의 ±10도·이동 4%·확대 ±5% affine + 아주 약한 밝기/대비/채도(hue 0). RandomErasing 은 쓰지 않습니다.
- 학습이 끝나면 `[안정성]`(best vs 마지막 5 epoch 평균), 후보×TTA 표, 정답 조합별 오류가 나옵니다.

```
!python -u compare.py model_v1_baseline:ckpt/A.pt \
    model_v1_tta:ckpt/A.pt:hv \
    model_v1_baseline:ckpt/G3.pt  model_v1_baseline:ckpt/G3_ema.pt  model_v1_baseline:ckpt/G3_swa.pt \
    model_v1_tta:ckpt/G3_ema.pt:hv \
    model:ckpt/V2m_ema.pt:on
```
(첫 항목 = 기준 A. 각 후보의 평균/crop/health F1, A 대비 차이와 95% 구간, 기준보다 좋은 비율, 예측이 달라진 장수, 파라미터/MB/CPU 시간)

### 조건부 실험 (위 결과에서 Apple/Grape recall 이 낮을 때만)
```
!python -u train.py --name G3_cw25  --aug mild --ema 0.998 --epochs 32 --cw_power 0.25 --crop_w 1.25
!python -u train.py --name G3_ls03  --aug mild --ema 0.998 --epochs 32 --label_smoothing 0.03
!python -u train.py --name G3_wd03  --aug mild --ema 0.998 --epochs 32 --wd 0.03
!python -u train.py --name G3_blur  --aug mild --ema 0.998 --epochs 32 --blur_p 0.1
```
한 번에 하나만 바꿉니다. Apple/Grape recall 이 오르면서 평균 F1 도 오를 때만 채택하세요 (precision 이 깎이는지는 `compare.py` 하단 per-class F1 로 확인).

### 안정성 확인 (최종 후보 설정에만)
```
!python -u train.py --name G3_s123  --aug mild --ema 0.998 --epochs 32 --seed 123
!python -u train.py --name G3_s2026 --aug mild --ema 0.998 --epochs 32 --seed 2026
```
seed 간 평균 F1 차이가 후보 간 차이와 비슷하면 그 설정들은 구분할 수 없으므로 더 튜닝하지 말고 가장 안정적인 쪽(EMA/SWA, 낮은 gap)을 고르세요.

## 3. 채택 규칙 (hidden test 기준)

1. **기준 A 보다 valid 가 낮아져도 채택할 수 있는 경우**: valid 차이가 부트스트랩 구간 안(0 포함)이면서, `[안정성]` 의 best-마지막평균 gap 이 A 보다 작거나 EMA/SWA 후보인 경우. (single-best 보다 비슷하거나 약간 낮아도 더 안정적이면 후보)
2. **valid 가 올라도 기각하는 경우**: 95% 구간에 0 이 포함되고 P(개선) < 0.8 인데 구조/학습 변경이 큰 경우, 또는 health/crop 한쪽만 오르고 다른 쪽이 내려간 경우.
3. SWA 는 **같은 run 의 마지막 5 epoch** 만 평균하고 BN 통계를 다시 계산합니다. 서로 다른 초기값의 모델끼리는 평균하지 않습니다. 파라미터 수와 state_dict 키는 그대로입니다.
4. TTA 는 재학습 없이 항상 후보가 될 수 있지만 **채점 시간**을 확인하세요 (`eval_tta.py`/`compare.py` 의 CPU 초, `make_submission.py --check`). 4뷰는 원본 대비 약 4배입니다.
5. train+valid 재학습(`--include_valid`), 강한 class weight, strong 증강, 용량 증가는 이번 단계에서 하지 않습니다.

## 4. 제출 후보 3개 (측정 후 숫자를 채우세요)

| 후보 | 구성 | valid 평균 / crop / health | A(0.9848/0.9789/0.9908) 대비 | hidden test 에서 좋아질 것으로 보는 이유 | 위험 요인 |
|---|---|---|---|---|---|
| 1 | A.pt + 가장 좋은 flip TTA | 측정 전 | 측정 전 | 재학습이 없어 A 의 일반화를 그대로 유지하면서, A 가 학습한 뒤집기 불변성으로 경계 샘플의 예측 분산을 줄임 | 이득이 작을 수 있음(노이즈 수준), 채점 시간 증가 (2뷰 ×2, 4뷰 ×4) |
| 2 | A 구조 + mild 증강 + EMA (G3 `_ema`, 필요하면 TTA) | 측정 전 | 측정 전 | 증강이 배경/조명/자세 차이에 대한 일반화를 늘리고, EMA 가 best-epoch 선택 편향을 없앰 | valid 가 A 보다 조금 낮을 수 있음 (선택 편향이 빠지는 만큼), 증강이 병반 단서를 약간 흐릴 가능성 |
| 3 | SWA(`G3_swa`) 또는 V2m 중 더 안정적인 것 | 측정 전 | 측정 전 | SWA: 한 epoch 에 의존하지 않고 평평한 해에 가까움. V2m: skip connection/SE/GAP+GMP 로 구조적 일반화 | SWA: 마지막 5 epoch 가 cosine 끝이라 이득이 작을 수 있음. V2m: 구조 변경이 커서 분산이 크고 채점 시간이 TTA 로 늘 수 있음 |

제출 순서 제안: 먼저 후보 1 (비용 0), 그다음 2, 마지막에 3. 하루 제출 수를 아끼려면 `compare.py` 에서 A 대비 P(개선) 이 낮은 후보는 제출하지 마세요.

## 5. 제출 폴더 만들기와 점검

```
!python make_submission.py --module model_v1_tta      --weights ckpt/A.pt       --tta hv --out submit_A_hv     --check
!python make_submission.py --module model_v1_baseline --weights ckpt/G3_ema.pt           --out submit_G3_ema   --check
!python make_submission.py --module model             --weights ckpt/V2m_ema.pt --tta on --out submit_V2m_ema --check
```
저장소의 `model.py` 는 건드리지 않고 후보마다 `<out>/model.py` + `<out>/best_model.pt` 를 만들고, 수정하지 않은 `check_submission.py` 를 그대로 실행합니다. 리더보드에는 그 폴더의 두 파일을 올리세요. `check_submission.py` 가 출력하는 valid 점수는 `compare.py` 와 같아야 합니다 (다르면 제출하지 마세요).