# CGH strict와 pivot 직접학습: 완료 결과와 영어 정보 효과

분석 대상은 2026-11-26 fresh cohort의 **seed 1·2**입니다. GRPO·split·strict·pivot은 각각 두 seed의 100-step 학습과 최종 평가가 완료됐습니다. Legacy·noIS는 seed 1만 완료돼 총 10회가 완료됐습니다. 전체 수치·진행은 [최신 결과](latest/RESULTS_KO.md), [진행 스냅샷](latest/PROGRESS_KO.md), [seed 직접 대조](latest/SEED_CONSISTENCY_KO.md), 문항별 기록·출처는 [CSV](latest/completed_items.csv)와 [JSON](latest/completed_results.json)에 있습니다.

## 두 seed에서 반복된 관측 (2026-11-26 16:42 KST)

| 조건 | Seed | Bengali /60 | Telugu /60 | Thai /60 | 전체 /180 | 같은 seed GRPO 대비 |
|---|---:|---:|---:|---:|---:|---:|
| Strict | 1 | 19 | 5 | 44 | 68, 37.78% | +7.22 pp |
| Strict | 2 | 23 | 3 | 42 | 68, 37.78% | +7.22 pp |
| Pivot | 1 | 21 | 12 | 26 | 59, 32.78% | +2.22 pp |
| Pivot | 2 | 21 | 16 | 23 | 60, 33.33% | +2.78 pp |

두 seed에서 GRPO·split은 모두 55/180이고 strict는 모두 68/180입니다. Strict의 기준선 대비 +13개가 반복됐습니다. Strict seed 2의 조건부 문항 bootstrap 구간은 GRPO 대비 [+2.22, +12.22] pp, split 대비 [+2.78, +11.67] pp입니다. 학습 seed 변동과 다중 비교 보정을 포함하지 않는 탐색적 구간입니다.

Strict−pivot은 seed 1 +5.00 pp, seed 2 +4.44 pp이지만 조건부 구간은 각각 [−2.22, +12.22], [−3.33, +12.22] pp로 모두 0을 포함합니다. 따라서 두 seed의 전체 점추정 방향은 일치해도 reference distillation 대비 우위를 확정하지 않습니다. 세 seed 평균은 아직 보류합니다.

언어별로 strict의 GRPO 대비 이득은 seed 1에서 +4/+2/+7개, seed 2에서 +5/+1/+7개입니다. 두 seed 모두 세 언어에서 증가했지만 Telugu 증가는 각각 2개·1개로 작습니다. “가장 약한 언어에서 가장 많이 개선한다”는 주장을 뒷받침하지 않습니다.

Pivot의 GRPO 대비 언어별 차이는 seed 1에서 +6/+9/−11개, seed 2에서 +3/+14/−12개입니다. Bengali·Telugu 증가와 Thai 감소가 반복됐습니다. Strict−pivot의 언어별 차이는 −2/−7/+18 → +2/−13/+19개로, Telugu에서 pivot이 높고 Thai에서 strict가 높은 방향은 같지만 Bengali 방향은 뒤집힙니다. 모든 언어의 결과가 일관되거나 strict가 균일하게 우수하다고 쓰지 않습니다.

## 1. Seed 1의 전체 조건 비교

Target은 숫자 정답과 target script 기준을 함께 통과한 결과입니다. 세 언어는 동일한 **60개 원문 문제의 번역**이므로 180개 독립 문제로 세지 않습니다.

| 조건 | Bengali /60 | Telugu /60 | Thai /60 | 전체 /180 | GRPO 대비 |
|---|---:|---:|---:|---:|---:|
| GRPO (`vanilla`) | 15 | 3 | 37 | 55, 30.56% | 기준 |
| `split` | 15 | 1 | 39 | 55, 30.56% | 0.00 pp |
| `cgh_strict` | 19 | 5 | 44 | 68, 37.78% | +7.22 pp |
| `pivot_rsft` | 21 | 12 | 26 | 59, 32.78% | +2.22 pp |
| `cgh_legacy` | 23 | 6 | 42 | 71, 39.44% | +8.89 pp |
| `cgh_strict_nois` | 26 | 7 | 39 | 72, 40.00% | +9.44 pp |

추가로 완료된 seed 2에서도 GRPO와 split은 각각 55/180입니다. 두 seed의 총점이 같다는 관측은 정규화 효과가 없다는 증명이나 동등성 검정이 아닙니다. Strict·pivot 직접 비교는 두 seed까지 완료됐습니다. Legacy·noIS 직접 비교는 seed 1에 한정됩니다. 사전 계획한 세 seed가 모두 완료되기 전 방법별 평균은 보류합니다.

| 직접 비교, seed 1 | 정답 수 차이 | 정확도 차이 | 조건부 95% 구간, pp |
|---|---:|---:|---:|
| strict − GRPO | +13 | +7.22 pp | [+1.67, +12.78] |
| strict − split | +13 | +7.22 pp | [+1.67, +12.78] |
| pivot − GRPO | +4 | +2.22 pp | [−5.00, +9.44] |
| pivot − split | +4 | +2.22 pp | [−5.56, +10.00] |
| strict − pivot | +9 | +5.00 pp | [−2.22, +12.22] |
| noIS − pivot | +13 | +7.22 pp | [+0.56, +13.89] |
| strict − noIS | −4 | −2.22 pp | [−7.78, +3.33] |
| strict − legacy | −3 | −1.67 pp | [−7.22, +4.44] |

구간은 원문 ID 60개를 20,000회 resample하고 세 번역을 함께 유지한 percentile bootstrap입니다(RNG 3213). 고정된 학습 seed의 두 모델에 대한 **문항 불확실성만** 반영합니다. 학습 seed 변동과 다중 비교 보정을 포함하지 않는 탐색적 구간이므로 최종 유의성 판정으로 쓰지 않습니다.

## 2. 영어 정보 효과는 어디까지 구분되는가

| 비교 | 실제로 바꾸는 것 | 현재 판단과 한계 |
|---|---|---|
| GRPO → split | 기존 응답의 정규화 방식 | 두 완료 seed에서 전체 target 차이 0. 정규화만으로 strict의 +13개를 재현하지는 못했지만 일반적인 무효과는 증명하지 못함 |
| split → strict | 원래 auxiliary 계수를 유지하면서 all-wrong 그룹의 positive pivot 응답에 weighted target 항 추가 | 두 seed 모두 +13개. 추가 응답으로 auxiliary 계수를 다시 계산하지 않는 구현에서도 개선 관측 |
| pivot → strict | 동일 영어 reference conditioning의 응답을 쓰되, 항상 positive 직접학습 대신 all-wrong 선택과 clipped/root weighting 적용 | seed 1 +9개, seed 2 +8개지만 두 구간 모두 0 포함. Gate와 weighting이 함께 다름 |
| pivot → noIS | 동일 split 기본 업데이트와 weight 1을 두고 positive extra의 채택을 all-wrong 그룹으로 제한 | +13개. 선택적 업데이트에 유리한 탐색적 관측이나 채택 수·총 gradient 크기·학습 경로까지 일치하는 비교는 아님 |
| noIS → strict | 동일 선택 규칙에 clipped/root weight 적용 | −4개. Weighting의 필요성·우위를 보여주지 못함. 해로움이 확정된 것도 아님 |
| legacy → strict | 역사적 coefficient 분기에서 strict 구성으로 전환 | −3개. 정규화·target/auxiliary 분기 등 여러 차이가 있어 auxiliary 효과 하나만 분리하는 ablation으로 부르지 않기 |

`pivot_rsft`는 **영어 질문을 붙여 얻은 정답 응답을 원래 질문 아래 직접 학습**하며, 원래 split 업데이트도 유지합니다. 기존의 group-centering pivot probe보다 강한 정보 접근 대조입니다. 단순히 “영어를 보여주기만 하면 같은 성능”이라고 단정할 수도, “영어 정보 효과를 제거했다”고 쓸 수도 없습니다.

완료 여섯 조건은 같은 초기 LoRA hash, training prompt stream, 190/60 분할, 모델 revision, 학습 engine 및 응답 수를 사용했습니다. 매 prompt마다 원래 응답 4개와 extra 2개를 생성합니다. GRPO·split도 같은 reference-conditioned extra를 생성하지만 업데이트에서는 버립니다. 따라서 비교는 **영어 보조 응답을 학습에 사용하는 방식**을 다룹니다. 생성 횟수는 같아도 생성된 응답 자체, token/FLOP, update 수, gradient 크기는 다릅니다.

영어 내용 자체의 순수 효과를 추정하려면 원래 질문만으로 extra를 생성하되 **같은 RSFT/선택 규칙·계수**를 적용한 대응 대조가 더 필요합니다. 원격 `q_resample`은 proposal과 업데이트를 함께 바꾸므로 그 하나로 순수 영어 효과를 식별하지 못합니다. **+7.22 pp를 “영어 효과 +2.22, estimator 효과 +5.00”으로 인과 분해하지 않습니다.**

### 언어별 차이가 결론을 바꾼다

Seed 1에서 strict − pivot의 순증가 9개는 Bengali **−2**, Telugu **−7**, Thai **+18**의 합입니다. Telugu는 pivot 12/60(20.00%), strict 5/60(8.33%)이고, Thai는 pivot 26/60(43.33%), strict 44/60(73.33%)입니다. Pivot은 GRPO보다 Telugu를 9개 늘렸지만 Thai를 11개 줄였습니다.

따라서 strict가 영어 직접학습보다 모든 언어에서 좋다는 주장, 특히 **가장 약한 언어에서 가장 많이 개선한다는 주장**은 현재 결과가 뒷받침하지 않습니다. 반대로 한 seed의 언어별 변화로 pivot의 일반적인 Thai 악화를 단정하지도 않습니다.

### 생성과 채택은 같은 예산이 아니다

| 조건, seed 1 | 생성한 extra | Positive extra | 채택 extra | Generated tokens |
|---|---:|---:|---:|---:|
| strict | 400 | 151 | 70 | 323,925 |
| pivot | 400 | 130 | 130 | 268,989 |
| noIS | 400 | 154 | 77 | 321,124 |

모두 총 응답 1,200개를 생성했습니다. Strict/noIS는 all-wrong일 때만 positive extra를 학습하고 pivot은 그룹 상태와 관계없이 positive extra를 학습합니다. 조건별 policy가 달라지므로 동일한 고정 응답 풀의 선택 결과가 아닙니다. 공유 GPU의 완료 시간이나 이 token 수만으로 계산 효율의 인과적 우위를 주장하지 않습니다.

## 3. 게재본에 반영할 수준

다음은 현재 완료 결과만 기술하는 잠정 문안입니다. 최종 게재본은 남은 seed와 외부 평가 결과를 취합한 후 갱신합니다.

> In two completed matched seeds, the strict construction obtained 68/180 language-gated correct responses in each seed, versus 55/180 for both scalar GRPO and separated normalization. Reference-assisted positive distillation obtained 59/180 and 60/180, with Bengali/Telugu gains and Thai losses relative to GRPO in both seeds. The strict–distillation difference remained uncertain under per-seed problem-cluster bootstraps. These exploratory observations do not establish the necessity of the deployed weighting or isolate the causal effect of English information; the unweighted selective variant obtained 72/180 in its first completed seed.

이론의 exact identity는 이상적 unfiltered sequence-ratio 항에 한정하고 실제 clipped/root·mean-token 구현은 biased surrogate로 명시합니다. 정확도 개선으로 exact gradient 복원을 입증할 수는 없습니다.

다음 판단은 사전 계획한 나머지 seed, 원격 후순위 학습 9회, 외부 2,069개 원문/4,138개 입력 평가를 기다립니다. 다른 서버의 실제 실행 상태는 여기서 관측하지 않았습니다. 소규모 holdout에서 잘 나온 조건만 최종 표에 고르는 대신 여섯 핵심 조건과 seed별 결과를 모두 보고합니다.
