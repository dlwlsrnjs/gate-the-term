# 완료 결과가 리뷰어 지적에 갖는 의미

2026-11-26 fresh cohort에서 핵심 여섯 조건의 seed 1, GRPO·split·strict·pivot seed 2까지 **10회**의 100-step 학습과 평가를 확인했습니다. `cgh_strict`·`pivot_rsft` seed 1도 완료됐습니다. 수치는 [최신 결과](latest/RESULTS_KO.md), 진행은 [진행 스냅샷](latest/PROGRESS_KO.md), 영어 정보 효과와 직접 대조는 [CGH·pivot 분석](CGH_PIVOT_KO.md)에 기록합니다. Pivot seed 2는 60/180(33.33%), strict seed 2는 68/180(37.78%)로 모두 완료됐습니다. [Seed 간 비교](latest/SEED_CONSISTENCY_KO.md)에서 최신 완료 여부를 확인합니다. 세 seed 평균은 아직 보류합니다.

## 현재 확인된 결과

동일 초기화와 prompt stream을 맞춘 seed 1의 target 정답 수는 GRPO **55/180**, split **55/180**, strict **68/180**, pivot RSFT **59/180**, legacy **71/180**, noIS **72/180**입니다. Target은 숫자 정답과 script 기준을 함께 요구합니다. 같은 60개 원문을 세 언어로 번역했으므로 180개 독립 문제로 세지 않습니다.

Seed 1에서 strict는 GRPO·split보다 +7.22 pp입니다. 원문 ID별 조건부 bootstrap 95% 구간은 두 비교 모두 [+1.67, +12.78] pp입니다. Strict − pivot은 +5.00 pp이나 구간 [−2.22, +12.22] pp에 0이 포함됩니다. 이 탐색적 구간은 학습 seed 변동과 다중 비교 보정을 포함하지 않습니다. 추가로 완료된 seed 2의 GRPO·split 총점도 각각 55/180이며 동등성을 입증하지는 않습니다.

Pivot seed 2는 bn/te/th 21/16/23개로 seed 1의 21/12/26개와 비교할 때 Telugu +4, Thai −3입니다. 대응 GRPO 대비 두 seed 모두 Bengali·Telugu 증가와 Thai 감소가 반복됐고 전체 이득은 +2.22→+2.78 pp입니다. Seed 2의 조건부 문항 구간 [−4.44, +10.00] pp는 학습 seed 변동을 포함하지 않습니다. 영어 보조 정보의 사용이 특히 Telugu 성능에 큰 영향을 줄 수 있음을 보여주는 관측이며, 순수 영어 내용의 인과 효과를 분리한 결과는 아닙니다.

Strict seed 2도 68/180으로 seed 1과 총점이 같고 대응 GRPO·split보다 각각 13개 높습니다. 언어별 정답은 19/5/44 → 23/3/42이며 GRPO 대비 이득은 +4/+2/+7 → +5/+1/+7개입니다. Seed 2의 조건부 문항 구간은 GRPO 대비 [+2.22, +12.22] pp, split 대비 [+2.78, +11.67] pp입니다. Strict−pivot은 seed 2 +4.44 pp, 구간 [−3.33, +12.22] pp로 여전히 불확실합니다. 두 seed에서 strict가 pivot보다 Telugu는 낮고 Thai는 높았으며 Bengali 차이의 방향은 뒤집혔습니다.

## 리뷰 지적별 판단

| 지적 | 지금 확인할 수 있는 내용 | 남은 비교 / 주장 제한 |
|---|---|---|
| Eq. 2의 정규화 변경이 개선 원인일 수 있음 | split은 두 완료 seed에서 GRPO와 총점 동일. strict는 두 seed 모두 split보다 13개 더 맞힘 | 정규화 무효과를 증명하지 않음. 남은 seed로 재현성과 변동 확인 |
| 추가 응답이 auxiliary 업데이트까지 바꿈 | 원래 auxiliary 계수를 유지한 strict에서도 두 seed에서 개선 관측 | legacy와 strict는 여러 coefficient 분기가 달라 auxiliary 효과 하나만 정량 분리한 실험은 아님 |
| 영어 reference 정보가 실질적인 원인일 수 있음 | 동일 reference conditioning의 정답을 원래 질문 아래 직접 학습하는 pivot RSFT 완료. 총점은 strict보다 낮지만 Telugu는 높음 | 강한 정보 접근 대조를 추가했으나 strict 우위는 아직 불확실. 순수 영어 내용 효과에는 동일 gate/objective의 q-only 대응 대조가 필요 |
| 선택적 gate / weighting의 필요성 | Seed 1에서 noIS는 72/180으로 strict 68/180보다 높음. noIS − pivot +7.22 pp, 조건부 구간 [+0.56, +13.89] | gate에 유리한 탐색적 관측이나 채택 수·gradient 크기·학습 경로도 달라짐. Weighting의 필요성이나 strict의 우위는 미입증 |
| 실제 estimator와 exact-gradient 이론이 다름 | 정확도 비교는 unbiasedness를 검증하지 않음 | exact identity는 이상적 unfiltered sequence-ratio 항에 한정. 실제 clipped/root/mean-token 구현은 biased surrogate로 기술 |
| 평가가 작고 seed 변동이 큼 | GRPO·split·strict·pivot 직접 대조는 원문 60개와 두 seed | 외부 2,069개 원문·4,138개 입력 평가 및 세 seed 완료 필요. 외부 평가도 모델·영역 다양성의 한계를 없애지는 않음 |
| 약한 언어에서 가장 크게 도움을 준다는 주장 | strict − pivot은 seed 1에서 −2/−7/+18, seed 2에서 +2/−13/+19개. Telugu의 strict 이득은 GRPO 대비 +2/+1개 | 보편적인 언어별 법칙을 주장하지 않기. 전체 차이가 Thai에 크게 좌우됨을 공개 |
| 더 단순한 수리 / whole-group gate와 비교 필요 | 원격 whole_gate·q_resample·component_zvp 각 세 seed 배정 유지 | 여기서는 다른 서버 실행 결과 미관측. adaptation을 full DAPO/RL-ZVP 재현으로 부르지 않기 |

## 진단 기여와 해석의 한계

여섯 조건의 seed 1에서 target 퇴화는 136–149/200 그룹(68.0–74.5%)인 반면 scalar zero-variance는 0–1/200(0–0.5%)입니다. Pivot에 scalar-zero 그룹 1개가 있으므로 “모든 run에서 scalar zero가 전혀 없다”고 쓰지 않습니다. 성분 수준의 contrast 부재를 scalar detector가 놓치는 진단 불일치는 다시 관측됐습니다.

이는 반복 문제와 변화하는 policy를 포함한 기술 통계입니다. 조건별 퇴화율 차이를 repair의 인과적 성공으로 읽지 않습니다. 관측 그룹에서 성분이 constant라는 사실은 population gradient가 사라진다는 뜻도 아닙니다.

영어 reference는 privileged training information입니다. 응답 생성 횟수·초기화·prompt stream을 맞췄어도 생성 token/FLOP, 채택 응답 수와 gradient 양은 같지 않습니다. 현재 비교는 영어 보조 응답의 사용 방식을 부분적으로 분리하며 strict의 전체 +7.22 pp를 영어 +2.22와 estimator +5.00으로 인과 분해할 수 없습니다.

## 게재본 반영

진단 기여를 중심으로 쓰고 strict의 기준선 대비 추가 개선은 현재 두 seed에서 반복된 관측으로 한정합니다. Seed 3과 외부 평가 전 일반적 우위나 안정성을 확정하지 않습니다. 강한 reference-distillation 대조를 추가했다는 점과 언어별 trade-off를 함께 보고합니다. “정확히 gradient를 복원한다”, “영어 정보 효과를 제거했다”, “가장 약한 언어를 가장 크게 개선한다”, “importance weighting이 필수다”라는 표현은 현재 결과에 맞지 않습니다.

새 cohort는 초기화·dropout·sampling·평가 크기를 맞춰 다시 실행했습니다. 기존 논문의 3-seed/6-seed 수치와 합치거나 직접 비교하지 않습니다. 최종 본문 표는 사전 계획한 세 seed를 취합한 뒤 작성하고 여섯 핵심 조건 전체와 언어별 결과를 부록에 남깁니다. 문안 예시는 [직접 대조 분석](CGH_PIVOT_KO.md)에 있습니다.
