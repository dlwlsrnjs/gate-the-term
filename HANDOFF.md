# 실행 인계 상태 (2026-09-20 KST)

- 원래 서버: real-model diagnostics 12 prompts × 16 draws 완료. 원래 LaTeX 표/PDF에 반영됨.
- 원래 서버: `vanilla_s1_n100` 현재 실행 중. 완료 결과로 간주하지 않음.
- 중복 방지를 위해 원래 **큐 관리자만 일시 정지**했음. 현재 vanilla 학습 자식 프로세스는 계속 실행됨. 다음 실험은 자동 시작하지 않음.
- 다른 서버 담당: 우선순위 1에서 seed 2,3의 다섯 방법 + seed 1의 vanilla 제외 네 방법 = 14 runs.
- smoke 두 종류는 원래 서버에서 통과했지만 새 서버에서도 smoke를 먼저 실행.
- 새 서버 학습은 처음부터 시작함. 중간 optimizer/RNG 복원은 지원하지 않음.

## 다른 서버 실행 명령 (설치·prepare·smoke 다음)

```bash
python -u scripts/run_suite.py --gpu 0 --seeds 2,3 --output outputs/controls > remote_seed23.log 2>&1
python -u scripts/run_suite.py --gpu 0 --seeds 1 --arms split,cgh_strict,pivot_rsft,cgh_strict_nois --output outputs/controls > remote_seed1.log 2>&1
```

두 GPU가 있으면 위 명령을 별도 터미널에서 두 번째만 `--gpu 1`로 실행해도 됨. 추가 절제 7 methods × 3 seeds는 이 14 runs 이후 README 5절에 따라 실행. 추가 실험은 원래 큐에서 아직 시작하지 않았음.

원래 vanilla seed1이 실패하면 해당 결과는 제외하고 새 서버에서 `--arms vanilla --seeds 1`을 실행해 전체 15 runs를 완성. 원래 결과를 회수할 때는 소스·환경·데이터·초기화 hash를 확인. 원래 모델 ID는 hub ID 형태지만 기존 캐시 snapshot `a09a35458c702b33eeacc393d103063234e8bc28`를 사용했으며 새 서버는 해당 snapshot 절대경로를 기록함.

## 원래 큐 재개 주의

원래 호스트의 `rebuttal_3213/remote_handoff.json`에 PID와 상태가 기록됨. 원격 담당 run을 취합하거나 큐에서 제외하기 전에 예전 큐를 재개하면 중복 실행함. PID는 재사용될 수 있으므로 명령 확인 없이 kill/signal 명령을 복사하지 말 것. 현재 자식 학습이 완료돼도 정지된 부모는 자동 표 갱신을 실행하지 않음. 결과를 확인해 원래 `export_latex.py`로 표를 갱신할 수 있음.

## GPU 여유에 따른 추가 로컬 배정

2026-09-20: GPU 1에 `component_zvp_s1_n100`을 추가로 시작했습니다. 원격 핵심 14개 배정은 유지합니다. **원격 추가 절제에서 component_zvp seed 1은 제외**하세요.

```bash
# entropy 대조실험의 남은 seeds만 원격 실행
python -u scripts/run_suite.py --gpu 0 --seeds 2,3 --arms component_zvp --output outputs/controls
# 나머지 추가 절제는 세 seeds 모두 원격 실행 가능
python -u scripts/run_suite.py --gpu 0 --seeds 1,2,3 --arms q_resample,absolute,whole_gate,cgh_legacy,primary_reward,dynamic_correctness --output outputs/controls
```

이 추가 배정도 시작 상태이며 완료 성능 결과는 아닙니다. GPU 1 결과가 완료되면 원래 서버의 LaTeX 표를 갱신하도록 실행했습니다.
