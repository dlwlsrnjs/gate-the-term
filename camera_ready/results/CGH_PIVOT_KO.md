# CGH·pivot: 3-seed 완료 결과와 영어 정보 효과

핵심 6조건 × 3 seeds의 18회 학습·fresh MGSM 평가와 외부 19개 모델 평가가 모두 완료됐습니다. 이 문서는 fresh MGSM 중심 분석이며, 외부 Indic-GSM8K·MSVAMP 결과와 최종 결론은 [최종 상세 결과](final_2026-11-27/RESULTS_KO.md)에 있습니다.

## 전체 성능과 직접 대조

| 조건 | Seed 1 /180 | Seed 2 /180 | Seed 3 /180 | 평균 target | GRPO 대비 |
|---|---:|---:|---:|---:|---:|
| GRPO | 55 | 55 | 55 | 30.56% | 기준 |
| split | 55 | 55 | 46 | 28.89% | −1.67 pp |
| strict | 68 | 68 | 66 | 37.41% | +6.85 pp |
| pivot RSFT | 59 | 60 | 53 | 31.85% | +1.30 pp |
| legacy | 71 | 71 | 65 | 38.33% | +7.78 pp |
| noIS | 72 | 69 | 69 | 38.89% | +8.33 pp |

| 직접 비교 | 평균 차이 pp | 탐색적 crossed 95% 구간 pp |
|---|---:|---:|
| strict − GRPO | +6.85 | [+2.59, +11.11] |
| strict − split | +8.52 | [+4.07, +13.15] |
| strict − pivot | +5.56 | [−1.11, +12.41] |
| noIS − pivot | +7.04 | [+0.56, +13.33] |
| strict − noIS | −1.48 | [−6.48, +3.52] |
| strict − legacy | −0.93 | [−5.00, +2.96] |

Strict의 GRPO 대비 차이는 세 seed에서 +7.22/+7.22/+6.11 pp로 모두 양수입니다. Split 대비도 +7.22/+7.22/+11.11 pp입니다. 첫 두 seed에서 0이었던 split−GRPO 차이는 seed 3에서 −5.00 pp가 돼 평균 −1.67 pp입니다. 따라서 “정규화만 바꾸면 항상 차이 0”으로 쓰지 않습니다. 정규화-only가 strict의 개선을 재현하지 못했다는 관측으로 한정합니다.

위 구간은 seed 3개와 원문 ID 60개를 독립적으로 resample하고 같은 원문의 세 번역을 함께 유지한 20,000회 crossed bootstrap입니다(RNG 3213). 기존 단일-seed 문항 구간과 다르며 학습 seed 변동도 경험적으로 포함합니다. 세 seed만으로 seed 분포를 추정하고 다중 비교 보정을 하지 않았으므로 탐색적 근거입니다. 540개 평가 결과는 540개 독립 문제가 아닙니다.

## 언어별 경향

| 조건 | Bengali 평균 | Telugu 평균 | Thai 평균 |
|---|---:|---:|---:|
| GRPO | 26.67% | 3.89% | 61.11% |
| strict | 35.00% | 6.11% | 71.11% |
| pivot | 32.78% | 22.78% | 40.00% |

Strict−GRPO의 언어별 정답 수 차이는 seed 1 +4/+2/+7, seed 2 +5/+1/+7, seed 3 +6/+1/+4개입니다. 세 언어의 방향은 모두 양수지만 Telugu 이득은 1–2개로 작습니다. 가장 약한 언어에서 가장 크게 개선한다는 주장은 뒷받침되지 않습니다.

Pivot−GRPO는 세 seed에서 Bengali +6/+3/+2개, Telugu +9/+14/+11개, Thai −11/−12/−15개입니다. Bengali·Telugu 증가와 Thai 감소가 세 seed 모두에서 반복됐습니다. Strict는 pivot보다 Telugu가 낮고 Thai가 높은 방향이 반복됐지만 Bengali 차이는 −2/+2/+4개로 뒤집힙니다. Strict가 영어 직접학습보다 모든 언어에서 우수하다고 쓰지 않습니다.

## 영어 정보·gate·weighting을 구분할 수 있는 범위

- `pivot_rsft`는 영어 reference를 붙여 얻은 positive 응답을 원래 질문 아래 직접 학습하며 원래 split 업데이트도 유지합니다. Group-centering probe보다 강한 정보 접근 대조입니다.
- Split→strict는 원래 auxiliary 계수를 보존하면서 all-wrong 그룹의 positive extra에 weighted target 항을 추가합니다. Strict 개선은 추가 응답으로 auxiliary 계수를 재계산하지 않는 구현에서도 세 seed에서 관측됐습니다.
- Strict→pivot 직접 대조는 영어 reference conditioning을 공유하지만 gate와 clipped/root weighting이 함께 다릅니다. 평균 strict 점수는 높아도 crossed 구간에 0이 포함돼 우위는 아직 불확실합니다.
- NoIS→pivot은 같은 split 업데이트와 weight 1에서 all-wrong 선택과 always-positive 사용을 비교합니다. NoIS가 세 seed 모두 높았고 평균 차이는 +7.04 pp입니다. 이는 선택적 업데이트에 유리한 관측이나 채택 수·총 gradient 크기·학습 경로까지 같지는 않습니다.
- NoIS는 세 seed 모두 strict보다 정답 수가 높았습니다(72>68, 69>68, 69>66). Weighting의 필요성·우위를 보여주지 못했습니다. 차이의 구간이 0을 포함하므로 weighting이 일반적으로 해롭다는 증명도 아닙니다.
- Legacy와 strict는 정규화·target/auxiliary 분기 등 여러 계수 구성이 다릅니다. 그 차이를 auxiliary 효과 하나만 분리한 ablation으로 부르지 않습니다.

여섯 조건은 동일 seed 안에서 초기 LoRA hash와 training prompt stream을 맞췄으며 같은 engine·190/60 분할·고정 모델 revision을 사용했습니다. 매 prompt에 원래 4개와 extra 2개, 각 run 1,200개 응답을 생성합니다. GRPO·split도 reference-assisted extra를 생성하되 업데이트에서는 버립니다. 응답 수가 같아도 실제 응답, token/FLOP, 채택 수, gradient 양은 같지 않습니다.

현재 대조는 영어 보조 응답의 **사용 방식**을 부분적으로 분리합니다. 영어 내용 자체의 순수 효과에는 같은 objective·gate를 유지하고 q-only proposal을 쓰는 대응 대조가 필요합니다. 원격 q_resample은 proposal과 업데이트를 함께 바꿉니다. Strict의 전체 +6.85 pp를 “영어 +1.30, estimator +5.56”으로 인과 분해하지 않습니다.

## 게재본에 반영할 잠정 문안

> Across three matched seeds, strict CGH reached 37.41% language-gated accuracy versus 30.56% for scalar GRPO and 28.89% for separated normalization. Reference-assisted positive distillation reached 31.85%, with Telugu gains and Thai losses in every seed. The strict–distillation difference remained uncertain under a crossed seed/problem bootstrap, while unweighted selective updates reached 38.89%. These exploratory results support evaluating selective updates separately from reference access, without establishing the necessity of weighting or isolating a pure English-content effect.

Exact identity는 이상적 unfiltered sequence-ratio 항에 한정하고 실제 clipped/root·mean-token 구현은 biased surrogate로 명시합니다. 정확도 개선은 unbiasedness나 exact gradient 복원을 입증하지 않습니다. 외부 2,069개 semantic IDs에서도 strict−split 개선은 반복됐지만, 한 base model·100 steps·산술 도메인이라는 제한은 유지합니다.
