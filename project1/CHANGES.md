# Project 1 개선 기록 (보고서용)

기준: 기존 모델(`model_v1_baseline.py`, 1,176,421개, valid 평균 Macro-F1 0.9489)
- crop Macro-F1 0.9266 (Apple 0.872 / Grape 0.945 / Tomato 0.963), health Macro-F1 0.9712

> 이 문서의 "기대 효과"는 일반적인 경험과 데이터 분포에 근거한 **예상**입니다. 코드 작성 환경에 GPU/PyTorch가 없어 실제 학습 결과는 아직 없습니다. 각 실험의 valid 점수로 확인해서 채택 여부를 정하세요.

## 1. 현재 병목

- 점수 = (crop + health) / 2 이므로 health(0.971)를 0.01 올리는 것보다 crop(0.927)을 0.01 올리는 쪽이 여유가 크다. crop 쪽에서 Apple F1이 가장 낮다.
- Apple 이 약한 이유(가설, 이미지 샘플과 분포 기준):
  1. 불균형: train 에서 Apple 16% vs Tomato 65%. 애매한 잎은 Tomato 로 쏠려 Apple recall 이 깎이고 F1 도 낮아진다.
  2. 시각적 유사성: Apple Healthy 와 Tomato Healthy 는 둘 다 매끈한 초록색 타원형 잎에 회색 배경이고, Apple 병반(갈색 점)은 Tomato 병반과 비슷해 보인다. 구분 단서는 잎 외곽 모양/톱니/잎맥 패턴이다.
  3. 구조: GAP 만 쓰면 국소 병반 신호가 희석되고, 기존 모델은 증강이 뒤집기뿐이라 방향/위치 변화에 약하다.
  4. 10 epoch 마지막에 최고점 → 아직 수렴 전이고 lr 감소가 없어서 후반 미세 조정이 안 됨.
- 검증 방법: 새 `train.py`는 마지막에 `[정답 조합별 오류 분석]`을 출력한다 (예: `Apple/Healthy ... 작물 0.90 (->Tomato 15)`). Apple 오류가 정말 Tomato 로 가는지, Healthy/Diseased 중 어느 쪽이 약한지 바로 확인할 수 있다.

## 2. 후보 비교와 선택

| 후보 | 상승 가능성 | 과적합 위험 | 구현 안정성 | 결정 |
|---|---|---|---|---|
| epoch 10 → 40~50 | 높음 | 낮음(증강/wd 병행) | 높음 | 채택 |
| Cosine + warmup | 높음 | 낮음 | 높음 | 채택 (OneCycle 은 `--sched onecycle` 로 비교 가능) |
| AdamW + weight decay 0.05 | 중간 | 낮춤 | 높음 | 채택 (BN/bias 는 decay 제외) |
| 클래스 가중 CE (빈도^-0.5) | 높음 (Macro-F1 직결) | 낮음 | 높음 | 채택 |
| crop/health loss 가중 (1.5 : 1) | 중간 | 낮음 | 높음 | 채택 |
| label smoothing 0.05 | 낮음~중간 | 낮춤 | 높음 | 채택 |
| 증강 medium (90도 회전, 약한 affine/색) | 높음 | 낮춤 | 높음 | 채택 |
| RandomErasing | 낮음~중간 | 낮춤 | 높음 | strong 에서만 사용 (병반 가림 위험이 있어 p=0.25, 작은 조각) |
| RandomResizedCrop | - | 병반/잎 모양 잘림 | 중간 | 보류 (모양이 Apple 구분 단서) |
| mixup/cutmix | - | 두 라벨 혼합, 병반 훼손 | 낮음 | 미채택 |
| WeightedRandomSampler | 중간 | 중복 샘플 과적합 | 높음 | 옵션 (`--sampler cell_sqrt`), 손실 가중치와 동시 사용 안 함 |
| Residual connection | 중간~높음 | 낮음 | 높음 | 채택 |
| SE attention | 낮음~중간 | 낮음 | 높음 | 채택 (가벼움) |
| Depthwise conv | 낮음 | - | 중간 | 미채택 (8천 장 규모에서 표준 conv 가 안전) |
| GAP + Global Max Pool 결합 | 중간 | 낮음 | 높음 | 채택 |
| head 분리 (MLP head 2개) | 낮음~중간 | 낮음 | 높음 | 채택 |
| capacity | - | - | - | 1.62M (한도 2M, 권장 1.8M 이하) |
| EMA | 중간 (안정화) | 낮음 | 높음 | 채택 (원본 vs EMA 를 매 epoch 같이 출력) |
| flip TTA | 낮음~중간 | 낮음 | 높음 | 채택 (2뷰, 끌 수 있음, 채점 시간만 주의) |
| train+valid 로 재학습 | 중간~높음 | valid 로 검증 불가 | 높음 | 마지막 단계에서만 (`--include_valid`) |

## 3. 실험 순서 (한 번에 하나씩 바꿈)

| 실험 | 모델 | 바뀌는 점 (이전 대비) | 검증하려는 것 |
|---|---|---|---|
| A | v1 구조 | 25 epoch + cosine + AdamW(wd 0.05), 증강 basic | 학습 설정만으로 얼마나 오르는가 |
| B | v1 구조 | A + 클래스 가중 CE(0.5) + crop_w 1.5 + label smoothing 0.05 | 불균형 대응이 Apple/Grape F1 을 올리는가 |
| C | v1 구조 | B + 증강 medium | 증강 효과 (과적합 감소) |
| D | v2 구조 | C 와 같은 설정에서 구조만 교체 (EMA 는 로그로, TTA 는 최종 로그로 비교) | 구조(Residual+SE+GMP+head 분리) 효과, EMA/TTA 이득 |
| P | v2 구조 | D + 증강 strong + 50 epoch | 최고 점수를 노리는 조합 (최우선 실행) |
| F | P 와 같은 설정 | `--include_valid` (train+valid 학습, 마지막 epoch 저장) | 데이터 28% 추가의 이득 (최종 제출 후보) |

EMA/TTA 효과는 같은 run 안에서 분리해 볼 수 있다: 매 epoch 줄에 원본(`avg F1`)과 EMA(`EMA ... avg`)가 같이 나오고, 마지막에 `[참고] tta=0/1` 비교가 나온다.

## 4. 변경 기록 (무엇을 / 왜 / 기대 효과)

**train.py**
- 학습 스케줄: 상수 lr → warmup 2 epoch + cosine (또는 OneCycle). 왜: 후반에 lr 을 낮춰 미세 조정해야 마지막 epoch 점수가 안정된다. 기대: 같은 epoch 수에서 F1 상승, 변동 감소.
- Adam → AdamW, wd 0.05 (BN/bias 제외). 왜: 파라미터 1.6M 대비 데이터 8천 장이라 규제가 필요. 기대: 과적합 감소.
- 클래스 가중 CE (빈도^-0.5, crop 과 health 각각). 왜: Macro-F1 은 소수 클래스(Apple, Grape, Healthy)의 recall 을 직접 반영한다. 기대: Apple/Healthy F1 상승. 완전 역수(1.0)는 precision 을 깎을 수 있어 제곱근을 사용.
- crop_w 1.5. 왜: crop 이 병목이고 공유 backbone 이 crop 을 더 신경 쓰게 한다. 기대: crop F1 소폭 상승, health 변화는 작아야 정상.
- label smoothing 0.05. 왜: 과신 예측 완화. 기대: 일반화 소폭 개선.
- EMA(0.998), 원본/EMA 중 valid 가 더 좋은 쪽을 best_model 로 저장. 왜: 마지막 epoch 의 흔들림 완화. 기대: +0.2~0.5%p 수준의 안정화.
- 혼합정밀도(AMP, GPU 에서만), 이미지 메모리 캐시, 매 epoch lr/train loss/valid loss/crop·health·평균 F1 출력, 클래스별 F1 약식 표시(A/G/T), 마지막에 조합별 오류 분석 출력, 로그 json 저장.
- 기본 저장 위치를 `ckpt/<name>.pt` 로 변경. 왜: 기존 `best_model.pt` 를 실수로 덮어쓰지 않게.
- `--model_module model_v1_baseline`: 이전 구조로도 같은 학습 코드를 돌릴 수 있게 (A, B, C 용).
- `--include_valid`, `--smoke`(코드 점검용 2 epoch), `--sampler cell_sqrt` 옵션.

**dataset.py**
- 증강 단계(`basic/medium/strong`): medium = 뒤집기 + 90도 회전 + RandomAffine(±20도, 이동 6%, 확대 0.9~1.1, 회색 채움) + 약한 ColorJitter(hue 0.02), strong = medium + RandomErasing(p=0.25). 왜: 잎은 방향이 제각각이라 회전/뒤집기에 안전하고, 색은 병반 단서라 hue 를 거의 안 건드린다. 기대: 과적합 감소, Apple 처럼 모양 단서가 중요한 클래스의 일반화 개선.
- 평가용 전처리(`train=False`)는 한 글자도 바꾸지 않았다 (ToTensor + Normalize(0.5, 0.5)).

**model.py (v2)**
- Residual block + SE, MaxPool → stride 2, 마지막 블록 두 번째 conv 는 1x1. 왜: 깊어져도 학습이 안정되고 채널 attention 으로 병반/모양 단서를 강조. 기대: 같은 파라미터 대비 표현력 증가.
- GAP + Global Max Pool 결합(512차원). 왜: 작지만 강한 병변 신호 보존.
- head 를 Dropout → Linear(512,128) → ReLU → Dropout → Linear 로 과제별 분리. 왜: 공유 특징 위에서 과제별 변환. `out_features` 속성 유지.
- flip TTA: `model.eval()` 일 때만 원본 + 180도 뒤집은 입력의 logits 평균 (파라미터/state_dict 변화 없음, `tta=False` 로 끌 수 있음). 기대: +0.1~0.3%p.
- 크기: 파라미터 1,615,237 + 버퍼 4,192 = 1,619,429 (한도 2,000,000, 권장 1,800,000 이하).

## 5. check_submission.py 통과 점검 (코드 수준)

| 항목 | 상태 |
|---|---|
| 클래스 이름 `MultiTaskCNN`, `MultiTaskCNN()` 인자 없이 생성 | 모든 인자에 기본값 |
| 출력 | (crop (B,3), health (B,2)) 튜플, train/eval 모드 모두 shape 추적으로 확인 |
| `crop_head.out_features == 3`, `health_head.out_features == 2` | 확인 |
| 파라미터 + float 버퍼 | 1,619,429 (한도 2M) |
| import | `torch`, `torch.nn`, `torch.nn.functional` 만 |
| model.py 크기 | 약 5KB (한도 200KB) |
| best_model.pt 크기 | 약 6.5MB 예상 (한도 10MB, 파라미터 1.62M x 4바이트) |
| state_dict | `torch.save(model.state_dict())` 만 사용, EMA 도 같은 키, `weights_only=True` 로 읽히는 순수 텐서 |
| 평가 전처리 | 변경 없음 |

한계: 이 환경에서는 PyTorch 를 설치할 수 없어서(네트워크 정책) 실제로 `check_submission.py` 를 돌리지 못했다. 가짜 torch 로 구조/파라미터 수/shape 만 확인했다(기존 v1 모델은 같은 방법으로 1,174,501 + 1,920 으로 정확히 재현됨). 학습 후 반드시 Colab 에서 `python check_submission.py` 를 실행하고, `time` 으로 소요 시간도 확인하세요 (TTA 때문에 v1 대비 채점 시간이 늘 수 있음).