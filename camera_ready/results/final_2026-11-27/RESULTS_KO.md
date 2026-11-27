# Camera-ready 추가실험 최종 결과

2026-11-27 12:04:09 KST에 배정된 학습·평가·최종 검증이 모두 끝났습니다. 실패한 Slurm 작업은 없으며, 최종 검증기는 base 1개와 primary adapter 18개를 합친 **19개 외부 평가 결과**를 확인했습니다.

## 산출물과 대용량 파일

- 이 디렉터리: 논문용 요약, LaTeX 표, per-run 지표·재현 설정, SHA256 목록
- [외부 평가 원시 예측 61개 파일 (고정 HF revision)](https://huggingface.co/jin-kwon/gate-the-term-camera-ready-20261126/tree/fb5bc2061369faed0ea464566ccb89b706788d1c/expanded-evaluation-20261127): 64,062,613 bytes
- [18개 primary adapter와 원래 학습 산출물 (고정 HF revision)](https://huggingface.co/jin-kwon/gate-the-term-camera-ready-20261126/tree/63cd7187e29bb27da0bcf238884ed0923c0682cc)
- HF 외부 평가 체크섬 manifest SHA256: `09aad27b9900dd866fb361373b88f25e45feb7322aa8ded0261000596cd57540`
- base model: `Qwen/Qwen2.5-7B-Instruct` revision `a09a35458c702b33eeacc393d103063234e8bc28`

GitHub에는 adapter와 `predictions.jsonl`을 중복 저장하지 않습니다. 원시 예측은 위 HF 고정 revision에, 집계값과 각 run의 adapter/code/data hash는 이 디렉터리에 둡니다.

## 완료 범위

| 단계 | 범위 | 상태 |
|---|---:|---|
| Primary 학습 | 6 arms × 3 seeds = 18 | 완료 |
| Secondary 학습 | 3 arms × 3 seeds = 9 | 완료 |
| Fresh MGSM holdout | 9 arms, 60 semantic IDs × 3 languages × 3 seeds | 완료 |
| 외부 base 평가 | 4,138 inputs | 완료 |
| 외부 adapter 평가 | 6 primary arms × 3 seeds × 4,138 inputs | 완료 |
| 최종 검증·집계 | 19/19 결과와 예측 개수 확인 | 완료 |

외부 평가는 사전 지정된 primary 6조건만 대상으로 합니다. Secondary 3조건은 fresh MGSM holdout까지 완료됐으며, 외부 4,138-input 평가 대상에는 포함되지 않았습니다.

## Fresh MGSM holdout

Target accuracy는 숫자 정답과 target script 조건을 함께 요구합니다. 원문 60개를 Bengali, Telugu, Thai로 평가했으며, 세 seed 평균입니다.

| Arm | Target accuracy | Script consistency | Vanilla 대비 |
|---|---:|---:|---:|
| vanilla | 30.56% | 99.26% | 기준 |
| split | 28.89% | 98.15% | −1.67 pp |
| cgh_strict | 37.41% | 99.63% | +6.85 pp |
| pivot_rsft | 31.85% | 100.00% | +1.30 pp |
| cgh_legacy | 38.33% | 99.81% | +7.78 pp |
| cgh_strict_nois | 38.89% | 100.00% | +8.33 pp |
| whole_gate | 35.56% | 99.26% | +5.00 pp |
| q_resample | 30.74% | 99.44% | +0.19 pp |
| component_zvp | 26.30% | 95.74% | −4.26 pp |

Strict−vanilla는 +6.85 pp, 탐색적 crossed 95% 구간은 [+2.59, +11.11] pp입니다. Strict−split은 +8.52 pp, [+4.07, +13.33] pp이며 세 seed에서 모두 양수였습니다. 반면 strict−pivot, strict−legacy, strict−noIS, strict−whole_gate 구간은 0을 포함합니다.

## 외부 평가: Indic-GSM8K

기존 MGSM과 정규화 영어 원문이 겹치는 250개를 제외한 1,069개 semantic IDs를 Bengali와 Telugu로 평가했습니다. 표는 세 seed 평균 language-gated accuracy입니다.

| Arm | 전체 | Bengali | Telugu |
|---|---:|---:|---:|
| base | 12.72% | 21.52% | 3.93% |
| vanilla | 14.03% | 24.42% | 3.65% |
| split | 14.17% | 24.66% | 3.68% |
| cgh_strict | 19.78% | 33.27% | 6.30% |
| pivot_rsft | 19.93% | 29.12% | 10.73% |
| cgh_legacy | 20.49% | 34.89% | 6.08% |
| cgh_strict_nois | 22.01% | 36.51% | 7.51% |

| 비교 | 평균 차이 | 탐색적 95% 구간 |
|---|---:|---:|
| strict − split | +5.61 pp | [+3.32, +8.34] |
| legacy − vanilla | +6.45 pp | [+2.90, +9.70] |
| strict − pivot | −0.14 pp | [−2.74, +2.76] |
| strict − noIS | −2.23 pp | [−4.44, −0.11] |
| strict − legacy | −0.70 pp | [−5.00, +2.48] |

정규화만 분리한 split은 vanilla와 사실상 같았고, strict는 split보다 높았습니다. 다만 pivot은 Telugu에서 strict보다 높았고, noIS는 이 데이터셋에서 strict보다 높았습니다. 따라서 Indic 결과는 선택적 target 개입의 효과를 지지하지만 strict weighting의 필요성은 지지하지 않습니다.

## 외부 평가: MSVAMP

MSVAMP의 1,000개 semantic IDs를 Bengali와 Thai로 평가했습니다. 공개 자료에 Telugu subset은 없습니다.

| Arm | 전체 | Bengali | Thai |
|---|---:|---:|---:|
| base | 57.90% | 51.80% | 64.00% |
| vanilla | 62.47% | 54.10% | 70.83% |
| split | 62.45% | 53.47% | 71.43% |
| cgh_strict | 66.37% | 59.67% | 73.07% |
| pivot_rsft | 58.77% | 54.63% | 62.90% |
| cgh_legacy | 66.60% | 60.00% | 73.20% |
| cgh_strict_nois | 65.33% | 60.20% | 70.47% |

| 비교 | 평균 차이 | 탐색적 95% 구간 |
|---|---:|---:|
| strict − split | +3.92 pp | [+2.38, +5.48] |
| legacy − vanilla | +4.13 pp | [+2.25, +6.08] |
| strict − pivot | +7.60 pp | [+5.25, +9.95] |
| strict − noIS | +1.03 pp | [−0.30, +2.38] |
| strict − legacy | −0.23 pp | [−1.73, +1.30] |

MSVAMP에서도 split은 vanilla와 같았고 strict는 split보다 높았습니다. Strict의 pivot 대비 차이는 이 데이터셋에서는 명확하지만, noIS·legacy와의 차이는 불확실합니다.

## 논문 주장에 대한 결론

새 결과는 다음의 좁은 주장을 뒷받침합니다.

1. scalar/group 수준의 퇴화와 component/term 수준의 유효 신호는 구분해야 합니다.
2. normalization 분리만 한 `split`은 개선을 재현하지 못했습니다.
3. 선택적 target 개입을 포함한 strict/legacy/noIS 계열은 fresh holdout뿐 아니라 Indic-GSM8K와 MSVAMP에서도 split·vanilla보다 대체로 높았습니다.
4. 외부 데이터셋에서도 개선이 반복돼 기존 60문제 holdout에만 맞춘 현상이라는 우려는 줄었습니다.

반면 다음은 주장하지 않습니다.

- clipped/root importance weighting이 필수이거나 항상 우수하다는 주장
- strict가 pivot RSFT보다 모든 데이터셋·언어에서 우수하다는 주장
- 실제 구현이 unbiased gradient 또는 exact gradient restoration을 제공한다는 주장
- 한 7B base model과 산술 데이터의 결과가 다른 모델·도메인으로 일반화된다는 주장

가장 안전한 게재본 결론은 **component-level signal starvation이라는 진단과 선택적 업데이트 방향은 강화됐지만, 특정 weighting을 포함한 strict estimator의 필요성은 입증되지 않았다**는 것입니다.

## 재현·검증 파일

- [`external_summary.json`](external_summary.json): 데이터셋·언어·arm별 외부 결과와 crossed 비교
- [`external_artifact_release.json`](external_artifact_release.json): HF revision·경로·크기·검증 상태
- [`external_table.tex`](external_table.tex): 외부 평가 LaTeX 표
- [`holdout_summary.json`](holdout_summary.json): 9조건 fresh MGSM 결과와 비교
- [`holdout_table.tex`](holdout_table.tex): fresh MGSM LaTeX 표
- [`per_run/`](per_run): 19개 평가의 `result.json`과 민감한 로컬 경로를 제거한 config
- [`checksums.sha256`](checksums.sha256): 이 디렉터리의 게시 파일 체크섬

모든 외부 평가는 greedy decoding, batch size 16, 최대 320 new tokens로 실행했습니다. 데이터 manifest SHA256은 `4f8ec84c8c21335443021b2677f5161a4e34da4aee484e4914de90fb4bdd82cd`, 평가 코드 SHA256은 `d2cd1702efd74addaff95a878d27ca06ca10e6899da7cee224fb8bc33f309f57`, scoring SHA256은 `fd97a360db191c7f9ac16aff369bcea93f1b0cd2c01c64d74d6489c36ed93a42`입니다.

## 통계·실험 제한

- 4,138개 언어별 입력은 2,069개 semantic IDs의 두 언어 버전이므로 모두 독립 문제가 아닙니다.
- crossed 구간은 seed와 semantic ID를 resample하고 번역을 묶은 탐색적 구간입니다.
- seed는 3개뿐이며 다중 비교 보정이 없습니다.
- response count는 맞췄지만 실제 token/FLOP, 채택 수, backward 비용은 완전히 같지 않습니다.
- 영어 reference 접근은 privileged information이며 pivot 대조가 이를 부분적으로만 통제합니다.
