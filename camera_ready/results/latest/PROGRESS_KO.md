# 실험 진행 현황

기준 시각: 2026-11-26T16:42:26+09:00

로컬: 완료 10/18, 실행 8, 대기 0, 실패 0.
로컬 마감: 2026-11-26T22:00:00+09:00. 다른 환경 배정은 후순위 학습 9회와 확장 평가 19개 모델입니다.
다른 환경의 실제 실행/완료 상태는 이 서버에서 확인하지 않았습니다.

## 실행 중

| 작업 | GPU | Step / 100 | 단계 | 평가 완료 언어 |
|---|---:|---:|---|---|
| cgh_legacy_s2_n100 | 1 | 75 | 학습 | — |
| cgh_strict_nois_s2_n100 | 1 | 64 | 학습 | — |
| vanilla_s3_n100 | 0 | 62 | 학습 | — |
| split_s3_n100 | 1 | 61 | 학습 | — |
| cgh_strict_s3_n100 | 0 | 60 | 학습 | — |
| pivot_rsft_s3_n100 | 0 | 62 | 학습 | — |
| cgh_legacy_s3_n100 | 1 | 14 | 학습 | — |
| cgh_strict_nois_s3_n100 | 0 | 3 | 학습 | — |

## 완료

- vanilla_s1_n100
- split_s1_n100
- pivot_rsft_s1_n100
- cgh_strict_nois_s1_n100
- vanilla_s2_n100
- split_s2_n100
- cgh_legacy_s1_n100
- cgh_strict_s1_n100
- pivot_rsft_s2_n100
- cgh_strict_s2_n100

## 대기 — 아직 로컬 예약


## 추가 원격 이관으로 명시적으로 해제된 작업

없음.

## 완료 시각 추정

| 작업 | 구분 | 예상 완료 KST |
|---|---|---|
| cgh_legacy_s2_n100 | running | 17:39 |
| cgh_strict_nois_s2_n100 | running | 17:58 |
| vanilla_s3_n100 | running | 18:04 |
| split_s3_n100 | running | 18:06 |
| cgh_strict_s3_n100 | running | 18:07 |
| pivot_rsft_s3_n100 | running | 17:59 |
| cgh_legacy_s3_n100 | running | 19:28 |
| cgh_strict_nois_s3_n100 | running | 19:50 |

예상은 최근 step 속도와 실제 완료 run의 후처리 시간에 따른 대기열 계산이며 보장 시각이 아닙니다. 18:05 이후 새 run을 시작하지 않고 22시에 이 실험의 로컬 작업을 종료하는 기존 설정을 유지합니다.

이 문서는 게시 시점 스냅샷입니다. 실제 이관 시에는 원래 서버의 최신 live_handoff.json에서 remote_fallback_jobs를 확인하고 pending 작업과 중복 실행하지 않습니다.
