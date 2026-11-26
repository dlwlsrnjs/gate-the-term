# 18회 완료 결과가 리뷰어 지적에 갖는 의미

로컬 핵심 6조건 × seeds 1,2,3의 학습과 최종 MGSM 평가가 모두 완료됐습니다. [3-seed 평균·crossed 구간](latest/THREE_SEED_RESULTS_KO.md), [per-seed 결과·출처](latest/RESULTS_KO.md), [영어 정보 효과 분석](CGH_PIVOT_KO.md), [완료 상태](latest/PROGRESS_KO.md)를 함께 확인합니다. 다른 환경의 후순위 학습 9회·확장 평가 19개 모델은 아직 취합하지 않았습니다.

## 주요 결과

평균 target 정확도는 GRPO 30.56%, split 28.89%, strict 37.41%, pivot 31.85%, legacy 38.33%, noIS 38.89%입니다. Target은 숫자 정답과 script 기준을 함께 요구합니다. 원문 60개를 세 언어·세 seed에서 평가한 것이며 540개 독립 문제가 아닙니다.

Strict−GRPO는 +6.85 pp, crossed 95% 구간 [+2.59, +11.11] pp이고 strict−split은 +8.52 pp, [+4.07, +13.15] pp입니다. 세 seed 모두 두 기준선보다 높았습니다. Strict−pivot은 +5.56 pp이나 [−1.11, +12.41] pp로 우위가 불확실합니다. 구간은 seed와 원문 ID를 각각 resample한 탐색적 구간이며, 세 seed에 기반하고 다중 비교 보정이 없습니다.

## 리뷰 지적별 판단

| 지적 | 완료 결과가 제공하는 근거 | 남는 주장 제한 |
|---|---|---|
| Eq. 2 정규화 변경이 개선 원인일 수 있음 | Split−GRPO는 0/0/−5.00 pp, 평균 −1.67 pp. Strict는 split보다 모든 seed에서 높음 | 정규화의 보편적 무효과를 증명한 것은 아님. 첫 두 seed의 차이 0만 강조하지 않기 |
| 추가 응답이 auxiliary 업데이트까지 바꿈 | 원래 auxiliary 계수를 보존한 strict에서도 세 seed 모두 GRPO·split 대비 개선 | Legacy−strict는 여러 계수 분기가 달라 auxiliary 효과만의 ablation은 아님 |
| 영어 reference가 실질적 원인일 수 있음 | 강한 pivot RSFT 대조를 세 seed 완료. Telugu 증가와 Thai 감소가 모든 seed에서 반복 | Strict−pivot의 구간은 0 포함. 같은 gate/objective의 q-only 대조 없이 순수 영어 효과를 분리했다고 주장하지 않기 |
| Gate·weighting의 필요성 | NoIS는 세 seed 모두 strict보다 높고 평균 38.89%. NoIS−pivot +7.04 pp, 구간 [+0.56, +13.33] | 선택적 업데이트에 유리한 관측이나 update 수·gradient 크기·학습 경로도 다름. Weighting의 필수성·우위는 미입증 |
| 구현과 exact-gradient 이론의 불일치 | 이번 정확도 대조는 unbiasedness를 검증하지 않음 | 이상적 identity와 실제 biased clipped/root·mean-token surrogate를 명확히 구분 |
| 평가가 작고 실험 범위가 좁음 | 원문 60개, 한 7B model, 100 steps, 세 seeds의 complete cohort 확보 | 외부 2,069개 원문·4,138개 입력 평가가 아직 미취합. 모델·영역 일반성은 여전히 제한 |
| 가장 약한 언어에서 가장 큰 개선 | Strict의 Telugu 평균 6.11% vs GRPO 3.89%, 각 seed 순증가 1–2개. Pivot은 Telugu 22.78%지만 Thai 40.00% | 보편적인 약한 언어 효과를 주장하지 않기. 언어별 상쇄를 전체 점수와 함께 공개 |
| 단순 repair·whole-group gate와 비교 | 원격 whole_gate·q_resample·component_zvp 각 세 seed 배정 유지 | 완료 미확인. Adaptation을 full DAPO/RL-ZVP 재현으로 부르지 않기 |

## 진단과 게재본 서술

성분 퇴화와 scalar zero-variance의 불일치 통계는 [전체 per-seed 결과](latest/RESULTS_KO.md)에 있습니다. 반복 문제와 변화하는 policy의 학습 그룹을 독립 표본으로 세지 않습니다. 관측 그룹에서 성분이 constant라는 사실을 population gradient의 소실로 확대하지 않습니다.

진단 기여와 strict의 기준선 대비 반복된 실용적 개선을 중심으로 쓰되, pivot 대비 우위와 weighting 필요성은 입증되지 않았다고 밝힙니다. “정확히 gradient 복원”, “영어 효과 제거”, “가장 약한 언어에서 최대 개선”이라는 표현은 현재 결과에 맞지 않습니다. 새 cohort를 역사적 3/6-seed 결과와 합치지 않습니다.

본문에는 여섯 조건의 3-seed 평균을 함께 제시하고, 부록에는 seed별·언어별 결과와 paired 차이·불확실성을 제공합니다. 외부 평가·후순위 비교가 수집되기 전 이를 완료된 결과로 게재하지 않습니다. 문안 예시는 [CGH·pivot 분석](CGH_PIVOT_KO.md)에 있습니다.
