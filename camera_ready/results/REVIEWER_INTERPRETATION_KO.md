# 완료 결과가 리뷰어 지적에 갖는 의미

이 문서는 2026-11-26의 fresh cohort에서 처음 완료된 `vanilla`와 `split`, seed 1을 해석합니다. 최신 문항별 수치·진행 상태는 [latest/RESULTS_KO.md](latest/RESULTS_KO.md)와 [latest/PROGRESS_KO.md](latest/PROGRESS_KO.md)를 확인합니다. 이후 완료되는 다른 조건의 결과를 이 두 모델의 관측으로 대신하지 않습니다.

## 현재 확인된 결과

동일 초기 LoRA hash, 동일 training prompt stream, 동일 60개 원문 ID × 세 언어에서 두 조건 모두 target 정답은 **55/180 = 30.56%**입니다. Target은 숫자 정답과 script 기준 통과를 함께 요구합니다.

| 언어 | Scalar GRPO | 정규화 분리만 적용 | 차이 |
|---|---:|---:|---:|
| Bengali | 15/60, 25.00% | 15/60, 25.00% | 0.00 pp |
| Telugu | 3/60, 5.00% | 1/60, 1.67% | −3.33 pp |
| Thai | 37/60, 61.67% | 39/60, 65.00% | +3.33 pp |
| 전체 | 55/180, 30.56% | 55/180, 30.56% | 0.00 pp |

같은 정답 수라도 예측이 같지는 않습니다. `split`은 GRPO 대비 8개 언어–문항 결과를 고치고 8개를 놓쳤습니다. Raw 숫자 정답은 55→57이지만, script 기준에서 제외되는 정답이 2개 생겨 target 총점은 같습니다. Script 기준 통과는 179/180→178/180입니다.

원문 ID를 묶은 단일-seed 조건부 bootstrap 95% 구간은 차이 기준 **[−4.44, +5.00] pp**입니다. 이는 학습 seed 불확실성을 포함하지 않고, 동등성 검정도 아닙니다. 세 seed 평균은 아직 보류합니다.

## 리뷰 지적별 판단

| 지적 | 지금 확인할 수 있는 내용 | 남은 비교 / 주장 제한 |
|---|---|---|
| Eq. 2의 정규화 변경이 개선 원인일 수 있음 | 정규화-only `split`의 첫 완료 seed에서는 전체 target 개선이 관측되지 않음 | 0.00 pp가 정규화 효과 부재의 증명은 아님. 세 seed의 `vanilla`–`split`과 `split`–`cgh_strict`가 필요 |
| 추가 응답이 auxiliary 업데이트까지 바꿈 | 완료 두 run의 초기화·prompt stream·응답 수와 학습 코드는 일치하며, 추가 응답은 이 두 조건의 update에 들어가지 않음 | 기존 CGH의 auxiliary 재계산 효과는 `cgh_legacy`–`cgh_strict`로 따로 판단 |
| 영어 reference 정보가 실질적인 원인일 수 있음 | 현재 두 run은 유효한 reference 학습 대조를 제공하지 않음 | 같은 pivot 정답을 원래 질문 아래 직접 학습하는 `pivot_rsft`와 `cgh_strict` 비교가 핵심. 현재 결과로 이 지적이 해소됐다고 쓰지 않기 |
| 실제 estimator와 exact-gradient 이론이 다름 | 정확도 비교는 unbiasedness를 검증하지 않음 | exact identity는 이상적 unfiltered sequence-ratio 항에 한정. clipped/root/mean-token 구현은 biased surrogate로 기술. `cgh_strict_nois`는 실용적 weighting 효과를 비교할 뿐 exactness를 증명하지 않음 |
| 평가가 작고 seed 변동이 큼 | 현재 관측은 원문 60개와 단일 seed의 두 조건 | 180개 독립 문제라고 세지 않기. 외부 2,069개 원문·4,138개 입력 평가 및 세 seed 완료를 기다리기 |
| 약한 언어에서 가장 크게 도움을 준다는 주장 | normalization-only 비교에서 Telugu는 2개 감소, Thai는 2개 증가 | CGH 결과도 아니고 언어 수·문항 수가 작음. 이 결과로 언어별 일반 법칙을 주장하지 않기 |
| 더 단순한 수리 / whole-group gate와 비교 필요 | 이 서버의 완료 두 run으로 판단할 수 없음 | 다른 환경에 배정한 `q_resample`, `component_zvp`, `whole_gate` × 세 seeds. adaptation을 full DAPO/RL-ZVP 재현으로 부르지 않기 |

## 진단 기여에는 어떤 의미가 있는가

GRPO에서는 200개 학습 그룹 중 146개(73%), split에서는 136개(68%)에서 target correctness가 퇴화했습니다. Scalar zero-variance는 두 run 모두 **0/200**입니다. 즉, 이번 실제 학습에서도 scalar detector가 관측하지 못하는 component-level contrast 부재가 나타났습니다.

이는 진단 불일치를 뒷받침하는 기술 통계입니다. 73%→68%를 유의한 감소나 repair의 성공으로 해석하지 않습니다. 그룹에는 반복 문제와 변화하는 policy가 포함됩니다. 성분이 관측 그룹 안에서 constant라는 사실은 population gradient가 사라진다는 뜻이 아닙니다.

## 게재본에 사용할 수 있는 잠정 표현

> In the first completed seed of the controlled cohort, separated normalization matched scalar GRPO on language-gated accuracy (55/180 for both), with offsetting Telugu and Thai changes. This preliminary comparison does not establish equivalence or identify the source of the historical CGH gain; the full seed cohort and reference-distillation controls remain necessary.

새 cohort의 수치를 기존 논문의 3-seed/6-seed 수치와 직접 합치거나 비교하지 않습니다. 이번에는 seeded initialization, dropout, sampling과 평가 크기를 맞춰 다시 실행했습니다. 원래 논문의 headline 개선을 새 단일-seed 대조로 입증하거나 반증하지 않습니다.

현재 단계의 핵심 메시지는 **진단 불일치는 다시 관측됐고, 정규화-only 개선은 첫 seed에서 관측되지 않았으며, CGH의 추가 이득과 원인은 아직 평가 중**이라는 것입니다.
