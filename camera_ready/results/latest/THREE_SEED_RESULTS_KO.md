# 로컬 핵심 18회 완료 — 3-seed 결과

기준 시각: 2026-11-26T20:23:14+09:00

핵심 6조건 × seeds 1,2,3의 100-step 학습과 최종 MGSM 평가가 모두 완료됐습니다. 다른 환경의 후순위 학습·확장 평가 결과를 포함하지 않습니다.

| 조건 | Seed 1 /180 | Seed 2 /180 | Seed 3 /180 | 3-seed 평균 target | GRPO 대비 | Raw 평균 | Script 통과 평균 |
|---|---:|---:|---:|---:|---:|---:|---:|
| vanilla | 55 | 55 | 55 | 30.56% | +0.00 pp | 31.11% | 99.26% |
| split | 55 | 55 | 46 | 28.89% | -1.67 pp | 29.81% | 98.15% |
| cgh_strict | 68 | 68 | 66 | 37.41% | +6.85 pp | 37.78% | 99.63% |
| pivot_rsft | 59 | 60 | 53 | 31.85% | +1.30 pp | 31.85% | 100.00% |
| cgh_legacy | 71 | 71 | 65 | 38.33% | +7.78 pp | 38.52% | 99.81% |
| cgh_strict_nois | 72 | 69 | 69 | 38.89% | +8.33 pp | 38.89% | 100.00% |

## 언어별 3-seed 평균

각 언어는 같은 원문 60개를 세 seed에서 평가했습니다. 180개 독립 문제를 뜻하지 않습니다.

| 조건 | Bengali | Telugu | Thai |
|---|---:|---:|---:|
| vanilla | 26.67% | 3.89% | 61.11% |
| split | 23.33% | 2.22% | 61.11% |
| cgh_strict | 35.00% | 6.11% | 71.11% |
| pivot_rsft | 32.78% | 22.78% | 40.00% |
| cgh_legacy | 37.22% | 8.89% | 68.89% |
| cgh_strict_nois | 40.56% | 11.11% | 65.00% |

## 짝지은 차이와 seed·원문 ID 불확실성

| 비교 | Seed 1 Δ pp | Seed 2 Δ pp | Seed 3 Δ pp | 평균 Δ pp | 탐색적 crossed 95% 구간 pp |
|---|---:|---:|---:|---:|---:|
| split − vanilla | +0.00 | -0.00 | -5.00 | -1.67 | [-6.11, +2.78] |
| cgh_strict − vanilla | +7.22 | +7.22 | +6.11 | +6.85 | [+2.59, +11.11] |
| pivot_rsft − vanilla | +2.22 | +2.78 | -1.11 | +1.30 | [-5.74, +8.15] |
| cgh_legacy − vanilla | +8.89 | +8.89 | +5.56 | +7.78 | [+2.96, +12.78] |
| cgh_strict_nois − vanilla | +9.44 | +7.78 | +7.78 | +8.33 | [+2.59, +14.07] |
| cgh_strict − split | +7.22 | +7.22 | +11.11 | +8.52 | [+4.07, +13.15] |
| pivot_rsft − split | +2.22 | +2.78 | +3.89 | +2.96 | [-3.70, +9.63] |
| cgh_strict − pivot_rsft | +5.00 | +4.44 | +7.22 | +5.56 | [-1.11, +12.41] |
| cgh_strict_nois − pivot_rsft | +7.22 | +5.00 | +8.89 | +7.04 | [+0.56, +13.33] |
| cgh_strict − cgh_strict_nois | -2.22 | -0.56 | -1.67 | -1.48 | [-6.48, +3.52] |
| cgh_strict − cgh_legacy | -1.67 | -1.67 | +0.56 | -0.93 | [-5.00, +2.96] |

20,000회 bootstrap에서 seed와 원문 ID를 독립적으로 resample하고, 같은 원문의 세 번역과 두 비교 조건을 함께 유지했습니다. 기존 단일-seed 문항 구간과 다릅니다. 3개 seed의 경험분포만 사용한 탐색적 구간으로 학습 변동 추정이 제한적이며 다중 비교 보정은 없습니다.

## 적용 범위

- Target은 숫자 정답과 script 기준을 함께 통과한 값입니다. Raw 정답·script 통과율은 별도 보고합니다.
- 원문은 60개이며 3언어·3seed 반복을 새로운 독립 문제로 세지 않습니다. 기존 논문의 역사적 cohort와 합치지 않습니다.
- 한 base model, 100 steps, privileged English reference access라는 제한을 유지합니다. 성능 비교는 exact gradient 복원이나 unbiasedness의 검증이 아닙니다.
- 다른 환경 배정인 학습 9회와 외부 2,069개 원문/4,138개 입력 × 19개 모델 평가는 이 로컬 완료와 별도입니다.

[문항별 CSV](completed_items.csv) · [원본 집계·출처](completed_results.json) · [3-seed JSON](three_seed_results.json) · [진행](PROGRESS_KO.md)

재생성: report_completed.py 실행 후 python camera_ready/report_three_seed.py를 실행합니다.
