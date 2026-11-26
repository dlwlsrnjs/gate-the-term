# 완료 실험 결과 — 잠정 분석

기준 시각: 2026-11-26T14:02:02+09:00

지정한 결과 폴더에서 확인한 fresh cohort의 완료 run만 포함합니다. 다른 서버 결과가 취합되지 않았다면 그 서버의 완료 여부는 미확인입니다. 단일 seed 관측은 방법의 최종 성능이나 세 seed 평균이 아닙니다.

| 조건 | Seed | Bengali | Telugu | Thai | 전체 target 정확도 | Raw 정답 | Script 기준 통과 |
|---|---:|---:|---:|---:|---:|---:|---:|
| split | 1 | 15/60 (25.00%) | 1/60 (1.67%) | 39/60 (65.00%) | 55/180 (30.56%) | 57/180 | 178/180 |
| vanilla | 1 | 15/60 (25.00%) | 3/60 (5.00%) | 37/60 (61.67%) | 55/180 (30.56%) | 55/180 | 179/180 |

## 사전 계획한 세 seed 평균

세 seed가 모두 완료된 방법만 평균을 표시합니다.

| 조건 | 이 폴더에서 완료 확인한 seeds | 3-seed target 정확도 |
|---|---|---:|
| vanilla | 1 | 미완료 — 보류 |
| split | 1 | 미완료 — 보류 |
| cgh_strict | 미완료 또는 미취합 | 미완료 — 보류 |
| pivot_rsft | 미완료 또는 미취합 | 미완료 — 보류 |
| cgh_legacy | 미완료 또는 미취합 | 미완료 — 보류 |
| cgh_strict_nois | 미완료 또는 미취합 | 미완료 — 보류 |
| whole_gate | 미완료 또는 미취합 | 미완료 — 보류 |
| q_resample | 미완료 또는 미취합 | 미완료 — 보류 |
| component_zvp | 미완료 또는 미취합 | 미완료 — 보류 |

Target 정확도는 수치 정답과 script 기준을 함께 요구합니다. Script 기준은 target script 비중 50% 이상인 결정적 규칙이며, 완전한 언어 식별이나 혼합 언어 부재를 보장하지 않습니다.

## 기준선과 문항별 비교

- Seed 1, split − GRPO: +0.00 pp; 개선 8개 / 악화 8개 언어–문항 결과. 원문 ID로 묶은 조건부 95% bootstrap 구간 [-4.44, +5.00] pp.

이 구간은 원문 60개를 resample하며 세 번역을 함께 유지합니다. 한 seed로 고정된 모델의 문항 변동만 반영하고 학습 seed 변동은 반영하지 않습니다. 전체 180개를 독립 문제로 취급하지 않습니다. 구간에 0이 포함된다는 이유로 두 방법의 동등성을 주장하지 않습니다.

## 학습 중 성분 퇴화 진단

| 조건 / seed | Target 퇴화 그룹 | All-wrong 그룹 | Scalar zero-variance 그룹 |
|---|---:|---:|---:|
| split / 1 | 136/200 | 123/200 | 0/200 |
| vanilla / 1 | 146/200 | 134/200 | 0/200 |

이는 각 run의 200개 학습 그룹에 대한 기술 통계입니다. 반복 문제와 변화하는 policy를 포함하므로 독립 표본의 효과 검정으로 읽지 않습니다.

## 첫 완료 비교(seed 1, vanilla/split)의 해석

현재 완료된 seed 1의 split은 GRPO와 target 정답 수가 같습니다. Bengali는 동일하고 Telugu는 2문제 감소, Thai는 2문제 증가했습니다. Raw 정답은 2개 늘었지만 script 기준을 함께 적용한 전체 target 정답 수는 늘지 않았습니다.

정규화 단독 변경의 전체 개선은 이 seed에서 관측되지 않았습니다. 그러나 이것이 정규화 효과가 없다는 증명이나 CGH 개선의 원인이 정규화가 아니라는 입증은 아닙니다. CGH, pivot 직접학습, weighting 및 다른 seed의 비교는 각각 별도의 결과로 판단해야 합니다.

성분 퇴화가 많은데 scalar zero-variance는 없다는 관측은 진단 불일치와 일치합니다. 이는 CGH의 성능 우위나 유일한 필요성을 입증하지 않습니다.

새 cohort는 초기화 순서, dropout, sampling과 평가 크기를 맞춰 다시 실행한 비교입니다. 기존 논문의 세 seed 수치와 직접 대조해 개선/악화를 판단하지 않습니다. 아직 세 seed가 끝나지 않은 방법 평균은 JSON에서 null로 남깁니다.

## 재현과 출처

문항별 수치는 completed_items.csv, paired ID·bootstrap·source SHA256은 completed_results.json에 있습니다. full response와 adapter는 원본 결과 폴더에 보존합니다.

```bash
python camera_ready/report_completed.py --root /path/to/camera_ready_experiments/runs \
  --state /path/to/camera_ready_experiments/status.json --output camera_ready/results/latest
```
