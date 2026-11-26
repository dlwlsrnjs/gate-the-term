# 완료 실험 결과 — 잠정 분석

기준 시각: 2026-11-26T20:23:14+09:00

지정한 결과 폴더에서 확인한 fresh cohort의 완료 run만 포함합니다. 다른 서버 결과가 취합되지 않았다면 그 서버의 완료 여부는 미확인입니다. 단일 seed 관측은 방법의 최종 성능이나 세 seed 평균이 아닙니다.

CGH와 pivot 직접학습의 상세 해석은 [직접 대조 분석](../CGH_PIVOT_KO.md)을 함께 확인합니다.

핵심 여섯 조건의 세 seed가 모두 완료됐습니다. [3-seed 평균·언어별 결과·crossed 구간](THREE_SEED_RESULTS_KO.md)을 확인합니다.

| 조건 | Seed | Bengali | Telugu | Thai | 전체 target 정확도 | Raw 정답 | Script 기준 통과 |
|---|---:|---:|---:|---:|---:|---:|---:|
| vanilla | 1 | 15/60 (25.00%) | 3/60 (5.00%) | 37/60 (61.67%) | 55/180 (30.56%) | 55/180 | 179/180 |
| split | 1 | 15/60 (25.00%) | 1/60 (1.67%) | 39/60 (65.00%) | 55/180 (30.56%) | 57/180 | 178/180 |
| cgh_strict | 1 | 19/60 (31.67%) | 5/60 (8.33%) | 44/60 (73.33%) | 68/180 (37.78%) | 69/180 | 179/180 |
| pivot_rsft | 1 | 21/60 (35.00%) | 12/60 (20.00%) | 26/60 (43.33%) | 59/180 (32.78%) | 59/180 | 180/180 |
| cgh_legacy | 1 | 23/60 (38.33%) | 6/60 (10.00%) | 42/60 (70.00%) | 71/180 (39.44%) | 71/180 | 180/180 |
| cgh_strict_nois | 1 | 26/60 (43.33%) | 7/60 (11.67%) | 39/60 (65.00%) | 72/180 (40.00%) | 72/180 | 180/180 |
| vanilla | 2 | 18/60 (30.00%) | 2/60 (3.33%) | 35/60 (58.33%) | 55/180 (30.56%) | 57/180 | 178/180 |
| split | 2 | 15/60 (25.00%) | 1/60 (1.67%) | 39/60 (65.00%) | 55/180 (30.56%) | 56/180 | 179/180 |
| cgh_strict | 2 | 23/60 (38.33%) | 3/60 (5.00%) | 42/60 (70.00%) | 68/180 (37.78%) | 69/180 | 179/180 |
| pivot_rsft | 2 | 21/60 (35.00%) | 16/60 (26.67%) | 23/60 (38.33%) | 60/180 (33.33%) | 60/180 | 180/180 |
| cgh_legacy | 2 | 23/60 (38.33%) | 6/60 (10.00%) | 42/60 (70.00%) | 71/180 (39.44%) | 71/180 | 180/180 |
| cgh_strict_nois | 2 | 22/60 (36.67%) | 5/60 (8.33%) | 42/60 (70.00%) | 69/180 (38.33%) | 69/180 | 180/180 |
| vanilla | 3 | 15/60 (25.00%) | 2/60 (3.33%) | 38/60 (63.33%) | 55/180 (30.56%) | 56/180 | 179/180 |
| split | 3 | 12/60 (20.00%) | 2/60 (3.33%) | 32/60 (53.33%) | 46/180 (25.56%) | 48/180 | 173/180 |
| cgh_strict | 3 | 21/60 (35.00%) | 3/60 (5.00%) | 42/60 (70.00%) | 66/180 (36.67%) | 66/180 | 180/180 |
| pivot_rsft | 3 | 17/60 (28.33%) | 13/60 (21.67%) | 23/60 (38.33%) | 53/180 (29.44%) | 53/180 | 180/180 |
| cgh_legacy | 3 | 21/60 (35.00%) | 4/60 (6.67%) | 40/60 (66.67%) | 65/180 (36.11%) | 66/180 | 179/180 |
| cgh_strict_nois | 3 | 25/60 (41.67%) | 8/60 (13.33%) | 36/60 (60.00%) | 69/180 (38.33%) | 69/180 | 180/180 |

## 사전 계획한 세 seed 평균

세 seed가 모두 완료된 방법만 평균을 표시합니다.

| 조건 | 이 폴더에서 완료 확인한 seeds | 3-seed target 정확도 |
|---|---|---:|
| vanilla | 1, 2, 3 | 30.56% |
| split | 1, 2, 3 | 28.89% |
| cgh_strict | 1, 2, 3 | 37.41% |
| pivot_rsft | 1, 2, 3 | 31.85% |
| cgh_legacy | 1, 2, 3 | 38.33% |
| cgh_strict_nois | 1, 2, 3 | 38.89% |
| whole_gate | 미완료 또는 미취합 | 미완료 — 보류 |
| q_resample | 미완료 또는 미취합 | 미완료 — 보류 |
| component_zvp | 미완료 또는 미취합 | 미완료 — 보류 |

Target 정확도는 수치 정답과 script 기준을 함께 요구합니다. Script 기준은 target script 비중 50% 이상인 결정적 규칙이며, 완전한 언어 식별이나 혼합 언어 부재를 보장하지 않습니다.

## 기준선과 문항별 비교

- Seed 1, split − GRPO: +0.00 pp; 개선 8개 / 악화 8개 언어–문항 결과. 원문 ID로 묶은 조건부 95% bootstrap 구간 [-4.44, +5.00] pp.
- Seed 1, cgh_strict − GRPO: +7.22 pp; 개선 21개 / 악화 8개 언어–문항 결과. 원문 ID로 묶은 조건부 95% bootstrap 구간 [+1.67, +12.78] pp.
- Seed 1, pivot_rsft − GRPO: +2.22 pp; 개선 27개 / 악화 23개 언어–문항 결과. 원문 ID로 묶은 조건부 95% bootstrap 구간 [-5.00, +9.44] pp.
- Seed 1, cgh_legacy − GRPO: +8.89 pp; 개선 25개 / 악화 9개 언어–문항 결과. 원문 ID로 묶은 조건부 95% bootstrap 구간 [+2.22, +15.56] pp.
- Seed 1, cgh_strict_nois − GRPO: +9.44 pp; 개선 28개 / 악화 11개 언어–문항 결과. 원문 ID로 묶은 조건부 95% bootstrap 구간 [+3.33, +16.11] pp.
- Seed 1, cgh_strict − split: +7.22 pp; 개선 22개 / 악화 9개 언어–문항 결과. 원문 ID로 묶은 조건부 95% bootstrap 구간 [+1.67, +12.78] pp.
- Seed 1, pivot_rsft − split: +2.22 pp; 개선 28개 / 악화 24개 언어–문항 결과. 원문 ID로 묶은 조건부 95% bootstrap 구간 [-5.56, +10.00] pp.
- Seed 1, cgh_strict − pivot_rsft: +5.00 pp; 개선 30개 / 악화 21개 언어–문항 결과. 원문 ID로 묶은 조건부 95% bootstrap 구간 [-2.22, +12.22] pp.
- Seed 1, cgh_strict_nois − pivot_rsft: +7.22 pp; 개선 30개 / 악화 17개 언어–문항 결과. 원문 ID로 묶은 조건부 95% bootstrap 구간 [+0.56, +13.89] pp.
- Seed 1, cgh_strict − cgh_strict_nois: -2.22 pp; 개선 11개 / 악화 15개 언어–문항 결과. 원문 ID로 묶은 조건부 95% bootstrap 구간 [-7.78, +3.33] pp.
- Seed 1, cgh_strict − cgh_legacy: -1.67 pp; 개선 11개 / 악화 14개 언어–문항 결과. 원문 ID로 묶은 조건부 95% bootstrap 구간 [-7.22, +4.44] pp.
- Seed 2, split − GRPO: +0.00 pp; 개선 8개 / 악화 8개 언어–문항 결과. 원문 ID로 묶은 조건부 95% bootstrap 구간 [-4.44, +4.44] pp.
- Seed 2, cgh_strict − GRPO: +7.22 pp; 개선 19개 / 악화 6개 언어–문항 결과. 원문 ID로 묶은 조건부 95% bootstrap 구간 [+2.22, +12.22] pp.
- Seed 2, pivot_rsft − GRPO: +2.78 pp; 개선 28개 / 악화 23개 언어–문항 결과. 원문 ID로 묶은 조건부 95% bootstrap 구간 [-4.44, +10.00] pp.
- Seed 2, cgh_legacy − GRPO: +8.89 pp; 개선 23개 / 악화 7개 언어–문항 결과. 원문 ID로 묶은 조건부 95% bootstrap 구간 [+3.33, +14.44] pp.
- Seed 2, cgh_strict_nois − GRPO: +7.78 pp; 개선 22개 / 악화 8개 언어–문항 결과. 원문 ID로 묶은 조건부 95% bootstrap 구간 [+1.67, +13.89] pp.
- Seed 2, cgh_strict − split: +7.22 pp; 개선 17개 / 악화 4개 언어–문항 결과. 원문 ID로 묶은 조건부 95% bootstrap 구간 [+2.78, +11.67] pp.
- Seed 2, pivot_rsft − split: +2.78 pp; 개선 30개 / 악화 25개 언어–문항 결과. 원문 ID로 묶은 조건부 95% bootstrap 구간 [-4.44, +10.00] pp.
- Seed 2, cgh_strict − pivot_rsft: +4.44 pp; 개선 32개 / 악화 24개 언어–문항 결과. 원문 ID로 묶은 조건부 95% bootstrap 구간 [-3.33, +12.22] pp.
- Seed 2, cgh_strict_nois − pivot_rsft: +5.00 pp; 개선 31개 / 악화 22개 언어–문항 결과. 원문 ID로 묶은 조건부 95% bootstrap 구간 [-2.78, +12.78] pp.
- Seed 2, cgh_strict − cgh_strict_nois: -0.56 pp; 개선 11개 / 악화 12개 언어–문항 결과. 원문 ID로 묶은 조건부 95% bootstrap 구간 [-6.11, +5.00] pp.
- Seed 2, cgh_strict − cgh_legacy: -1.67 pp; 개선 6개 / 악화 9개 언어–문항 결과. 원문 ID로 묶은 조건부 95% bootstrap 구간 [-5.57, +2.22] pp.
- Seed 3, split − GRPO: -5.00 pp; 개선 4개 / 악화 13개 언어–문항 결과. 원문 ID로 묶은 조건부 95% bootstrap 구간 [-9.44, -1.11] pp.
- Seed 3, cgh_strict − GRPO: +6.11 pp; 개선 18개 / 악화 7개 언어–문항 결과. 원문 ID로 묶은 조건부 95% bootstrap 구간 [+1.67, +10.56] pp.
- Seed 3, pivot_rsft − GRPO: -1.11 pp; 개선 26개 / 악화 28개 언어–문항 결과. 원문 ID로 묶은 조건부 95% bootstrap 구간 [-8.89, +6.67] pp.
- Seed 3, cgh_legacy − GRPO: +5.56 pp; 개선 17개 / 악화 7개 언어–문항 결과. 원문 ID로 묶은 조건부 95% bootstrap 구간 [+1.11, +10.00] pp.
- Seed 3, cgh_strict_nois − GRPO: +7.78 pp; 개선 29개 / 악화 15개 언어–문항 결과. 원문 ID로 묶은 조건부 95% bootstrap 구간 [+0.56, +15.00] pp.
- Seed 3, cgh_strict − split: +11.11 pp; 개선 25개 / 악화 5개 언어–문항 결과. 원문 ID로 묶은 조건부 95% bootstrap 구간 [+5.56, +16.11] pp.
- Seed 3, pivot_rsft − split: +3.89 pp; 개선 31개 / 악화 24개 언어–문항 결과. 원문 ID로 묶은 조건부 95% bootstrap 구간 [-3.33, +11.11] pp.
- Seed 3, cgh_strict − pivot_rsft: +7.22 pp; 개선 34개 / 악화 21개 언어–문항 결과. 원문 ID로 묶은 조건부 95% bootstrap 구간 [-0.00, +14.44] pp.
- Seed 3, cgh_strict_nois − pivot_rsft: +8.89 pp; 개선 33개 / 악화 17개 언어–문항 결과. 원문 ID로 묶은 조건부 95% bootstrap 구간 [+1.67, +16.67] pp.
- Seed 3, cgh_strict − cgh_strict_nois: -1.67 pp; 개선 15개 / 악화 18개 언어–문항 결과. 원문 ID로 묶은 조건부 95% bootstrap 구간 [-8.33, +5.00] pp.
- Seed 3, cgh_strict − cgh_legacy: +0.56 pp; 개선 8개 / 악화 7개 언어–문항 결과. 원문 ID로 묶은 조건부 95% bootstrap 구간 [-3.33, +4.44] pp.

이 탐색적 구간은 원문 60개를 resample하며 세 번역을 함께 유지합니다. 한 seed로 고정된 모델의 문항 변동만 반영하고 학습 seed 변동은 반영하지 않습니다. 다중 비교 보정은 하지 않았습니다. 전체 180개를 독립 문제로 취급하지 않습니다. 구간에 0이 포함된다는 이유로 두 방법의 동등성을 주장하지 않습니다.

## 학습 중 성분 퇴화 진단

| 조건 / seed | Target 퇴화 그룹 | All-wrong 그룹 | Scalar zero-variance 그룹 |
|---|---:|---:|---:|
| vanilla / 1 | 146/200 | 134/200 | 0/200 |
| split / 1 | 136/200 | 123/200 | 0/200 |
| cgh_strict / 1 | 142/200 | 127/200 | 0/200 |
| pivot_rsft / 1 | 143/200 | 129/200 | 1/200 |
| cgh_legacy / 1 | 142/200 | 126/200 | 0/200 |
| cgh_strict_nois / 1 | 149/200 | 133/200 | 0/200 |
| vanilla / 2 | 136/200 | 127/200 | 0/200 |
| split / 2 | 141/200 | 132/200 | 0/200 |
| cgh_strict / 2 | 134/200 | 122/200 | 0/200 |
| pivot_rsft / 2 | 129/200 | 115/200 | 0/200 |
| cgh_legacy / 2 | 131/200 | 118/200 | 0/200 |
| cgh_strict_nois / 2 | 134/200 | 119/200 | 0/200 |
| vanilla / 3 | 144/200 | 131/200 | 0/200 |
| split / 3 | 144/200 | 135/200 | 0/200 |
| cgh_strict / 3 | 135/200 | 122/200 | 0/200 |
| pivot_rsft / 3 | 132/200 | 122/200 | 1/200 |
| cgh_legacy / 3 | 148/200 | 130/200 | 0/200 |
| cgh_strict_nois / 3 | 140/200 | 125/200 | 0/200 |

## 추가 생성과 업데이트

| 조건 / seed | 추가 생성 중 positive 수 / 400 | 채택된 extra 수 | 채택 규칙 | Generated tokens |
|---|---:|---:|---|---:|
| vanilla / 1 | 148/400 | 0 | none | 327322 |
| split / 1 | 149/400 | 0 | none | 330358 |
| cgh_strict / 1 | 151/400 | 70 | positive-only | 323925 |
| pivot_rsft / 1 | 130/400 | 130 | positive-only | 268989 |
| cgh_legacy / 1 | 144/400 | 63 | contrast | 321907 |
| cgh_strict_nois / 1 | 154/400 | 77 | positive-only | 321124 |
| vanilla / 2 | 121/400 | 0 | none | 320361 |
| split / 2 | 120/400 | 0 | none | 319382 |
| cgh_strict / 2 | 114/400 | 40 | positive-only | 317794 |
| pivot_rsft / 2 | 118/400 | 118 | positive-only | 268042 |
| cgh_legacy / 2 | 123/400 | 47 | contrast | 319649 |
| cgh_strict_nois / 2 | 122/400 | 41 | positive-only | 315620 |
| vanilla / 3 | 129/400 | 0 | none | 331311 |
| split / 3 | 126/400 | 0 | none | 329619 |
| cgh_strict / 3 | 130/400 | 49 | positive-only | 324700 |
| pivot_rsft / 3 | 126/400 | 126 | positive-only | 280428 |
| cgh_legacy / 3 | 127/400 | 70 | contrast | 322884 |
| cgh_strict_nois / 3 | 134/400 | 57 | positive-only | 320511 |

positive-only 조건의 채택 수는 positive extra 업데이트 수입니다. legacy의 contrast 채택은 같은 의미가 아니므로 직접 동치로 비교하지 않습니다. 같은 생성 개수라도 token/FLOP, 채택 수, gradient 크기는 같지 않습니다. 조건별 policy가 달라지므로 동일 conditioning을 써도 생성 응답은 동일하게 고정되지 않습니다.

이는 각 run의 200개 학습 그룹에 대한 기술 통계입니다. 반복 문제와 변화하는 policy를 포함하므로 독립 표본의 효과 검정으로 읽지 않습니다.

## 해석 범위

성능과 영어 정보 효과의 비교는 [CGH·pivot 직접 대조](../CGH_PIVOT_KO.md), 각 리뷰 지적에 대한 판단은 [리뷰 해석](../REVIEWER_INTERPRETATION_KO.md)에 기록합니다. 이 문서는 완료된 수치를 자동 집계하며, 특정 한 seed의 해석을 이후 완료 결과에 대신 적용하지 않습니다.

성분 퇴화 그룹과 scalar zero-variance 그룹의 빈도 차이는 진단 불일치에 대한 기술 통계입니다. 퇴화율 감소 자체를 repair 성공으로 해석하거나, 정확도 개선으로 population gradient의 복원·unbiasedness를 주장하지 않습니다.

새 cohort는 초기화 순서, dropout, sampling과 평가 크기를 맞춰 다시 실행한 비교입니다. 기존 논문의 세 seed 수치와 직접 대조해 개선/악화를 판단하지 않습니다. 아직 세 seed가 끝나지 않은 방법 평균은 JSON에서 null로 남깁니다.

## 재현과 출처

문항별 수치는 completed_items.csv, paired ID·bootstrap·source SHA256은 completed_results.json에 있습니다. full response와 adapter는 원본 결과 폴더에 보존합니다.

```bash
python camera_ready/report_completed.py --root /path/to/camera_ready_experiments/runs \
  --state /path/to/camera_ready_experiments/status.json --output camera_ready/results/latest
```
