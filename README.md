# Gate the Term: 다른 서버에서 추가 대조실험 실행하기

이 저장소는 제출 논문의 역사적 결과를 그대로 재현하는 코드가 아니라, 동일한 초기화·샘플링 조건에서 reviewer 질문을 확인하는 **새 대조실험 묶음**입니다. 현재 7B smoke 2종만 원래 서버에서 통과했습니다. 이 저장소에 완료된 100-step 성능 결과는 없습니다. 아래 절차로 미완료 실험을 실행하고 결과를 원래 서버로 가져옵니다.

## 1. 서버·환경 준비

권장: Linux, Python 3.11, NVIDIA Ampere 이상 BF16 GPU. 원래 검증 장비는 H100 80GB입니다. 실행 전 **24,000 MiB 이상 여유**를 요구하지만 이는 예약도, 모든 GPU에서 OOM이 안 난다는 보장도 아닙니다. 모델·가상환경·결과용 디스크 공간도 확보하세요. 24GB 카드에서 여유 임계값을 임의로 낮추는 것보다 더 큰 GPU가 안전합니다. 양자화·짧은 출력·작은 모델로 바꾸면 별도 비교군으로 취급해야 합니다.

```bash
git clone https://github.com/dlwlsrnjs/gate-the-term.git
cd gate-the-term
bash scripts/setup.sh
source .venv/bin/activate
# 원하는 충분한 디스크 공간의 캐시 경로를 한 번 정하고 이후 동일하게 사용
export HF_HOME="$HOME/.cache/huggingface"
python scripts/prepare.py
python scripts/check_environment.py --gpu 0
python rebuttal_3213/check_controls.py
```

`setup.sh`는 원래 서버에서 성공한 torch 2.5.1+cu121 및 패키지 버전을 고정합니다. 설치 실패 시 버전을 몰래 바꾸지 말고 오류를 기록하세요. Python 3.11 명령이 다르면 `PYTHON_BIN=/path/to/python3.11 bash scripts/setup.sh`를 사용합니다. 다운로드는 인터넷이 필요합니다. `prepare.py`는 공식 MGSM 네 파일을 내려받아 기존 실험 데이터와 SHA256을 대조하고, Qwen2.5-7B-Instruct의 `a09a35458c702b33eeacc393d103063234e8bc28` revision을 받습니다. 학습은 준비된 로컬 모델만 사용합니다. 모델 캐시 절대경로는 `scripts/model_path.txt`에 자동 기록되므로 다른 서버의 이 파일을 복사하지 마세요.

데이터 출처: [Google Research MGSM](https://github.com/google-research/url-nlp/tree/main/mgsm). 모델: [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct). 각각의 원본 라이선스·사용 조건을 따릅니다. 원래 trainer는 helper 함수 제공 및 기록 보존용입니다. 아래 새 실행기를 사용하세요.

## 2. 먼저 smoke 실행

```bash
python scripts/run_suite.py --gpu 0 --arms cgh_strict,component_zvp \
  --seeds 3214 --steps 1 --eval-n 1 --output outputs/smoke
```

두 실험이 모두 `DONE`으로 끝나야 합니다. smoke는 각 언어 1문제만 평가하므로 성능 주장에 사용하지 않습니다. 원래 서버의 entropy smoke 1 step은 약 3분이었으며 모델 로딩·평가 시간은 별도입니다. 100 steps × 3 seeds × 여러 방법은 수십 시간 이상 걸릴 수 있습니다.

## 3. 우선순위 1: 5개 방법 × 3 seeds

**현재 인계에서는 [HANDOFF.md](HANDOFF.md)의 14-run 명령을 먼저 사용하세요.** 원래 서버에서 진행 중인 vanilla seed1과 중복되지 않도록 분담했습니다. 아래 전체 15-run 명령은 원래 결과를 쓰지 않고 새 서버에서 전부 실행할 때의 대안입니다.

먼저 정보 제공 효과와 정규화/선택적 업데이트 효과를 구분하는 15 runs를 완료하세요. 순서는 seed 안에서 다섯 방법을 비교하도록 고정했습니다.

| arm | 확인하는 질문 |
|---|---|
| `vanilla` | 동일 환경의 scalar GRPO 기준선 |
| `split` | component 정규화만으로 설명되는가? |
| `cgh_strict` | 원래 auxiliary 계수를 보존한 선택적 target 업데이트 |
| `pivot_rsft` | 동일 pivot의 정답 응답을 원래 질문에 distill하는 비교군 |
| `cgh_strict_nois` | clipped/root importance weight가 필요한가? |

```bash
# 접속이 끊겨도 계속 실행하려면 tmux 안에서 실행
# tmux new -s gate
python -u scripts/run_suite.py --gpu 0 --seeds 1,2,3 \
  --output outputs/controls > outputs_primary.log 2>&1
```

다른 터미널에서:

```bash
tail -f outputs_primary.log
cat outputs/controls/status_gpu0.json
tail -f outputs/controls/vanilla_s1_n100.log
nvidia-smi
```

CUDA_VISIBLE_DEVICES를 별도로 설정하지 마세요. `--gpu`는 nvidia-smi의 물리 GPU 번호이며 실행기가 trainer에 넘깁니다. GPU 한 장에는 이 실행기 한 개만 사용하세요. 여유 부족·오류 시 자동으로 다음 실험을 넘기지 않고 멈춥니다. 문제를 해결하고 같은 명령을 다시 실행하세요.

## 4. GPU / 서버가 여러 대면 seed 단위 분담

모든 서버에서 같은 커밋·동일 버전으로 설치하고 각각 `prepare.py`를 실행합니다. **seed를 겹치지 않게 배정**하세요. 서로 다른 서버 사이에는 자동 분산 잠금이 없습니다.

```bash
# 서버 A 또는 GPU 0
python -u scripts/run_suite.py --gpu 0 --seeds 1 --output outputs/controls > seed1.log 2>&1
# 서버 B, 또는 같은 서버의 GPU 1 (별도 터미널)
python -u scripts/run_suite.py --gpu 1 --seeds 2 --output outputs/controls > seed2.log 2>&1
# 서버 C의 GPU 0 (별도 서버이므로 GPU 번호는 0 가능)
python -u scripts/run_suite.py --gpu 0 --seeds 3 --output outputs/controls > seed3.log 2>&1
```

GPU 1이 없는 서버 B에서는 `--gpu 0`을 사용하세요. 같은 서버·같은 output root에서는 GPU/작업별 잠금으로 중복 실행을 막습니다. seed별 모든 비교 방법의 초기 LoRA hash가 같아야 합니다. 이미 학습된 CGH adapter에서 다른 arm을 시작하지 않습니다.

원래 서버의 로컬 큐 상태는 [HANDOFF.md](HANDOFF.md)를 확인하세요. 여기 안내의 분담은 자동 서버 예약이 아닙니다.

## 5. 우선순위 2: 요청에 대응하는 작은 절제/기준선

먼저 1순위 15 runs를 확보하세요. 시간이 남으면 같은 seed·steps·평가 조건으로 아래 명령을 실행합니다.

```bash
python -u scripts/run_suite.py --gpu 0 --seeds 1,2,3 \
  --arms component_zvp,q_resample,absolute,whole_gate,cgh_legacy,primary_reward,dynamic_correctness \
  --output outputs/controls > extra_controls.log 2>&1
```

| arm | 의미 / 해석 제한 |
|---|---|
| `component_zvp` | 퇴화한 target component에 detached token-entropy 항을 적용한 adaptation. 원 논문 전체 stack 재현 아님 |
| `q_resample` | 추가 2개를 영어 pivot 없이 원래 질문에서 생성 |
| `absolute` | 퇴화 그룹에 absolute signed target signal 적용 |
| `whole_gate` | 퇴화 시 원래 auxiliary 업데이트까지 제거하는 비교 |
| `cgh_legacy` | 옛 coefficient 분기를 새 공통 engine에서 실행. 역사적 실행의 bitwise 재현 아님 |
| `primary_reward` | correctness를 곱해 보조 보상을 gating한 scalar reward |
| `dynamic_correctness` | 고정 추가 2개 예산에서 correctness contrast filtering. full DAPO 아님 |

`cgh_noparallel`도 지원하지만 일반적인 검증 지시를 쓰는 별도 exploration입니다. 기본 실행에서 제외했습니다. 반박에는 reviewer 질문에 직접 대응하는 사소한 추가 실험만 선택하고, 광범위한 새로운 모델·도메인 실험으로 범위를 확장하지 마세요.

## 6. 실패 / 이어서 실행의 정확한 의미

- 완성된 `result.json`과 config가 맞는 run만 건너뜁니다.
- 미완료 디렉터리가 있으면 중단합니다. Adam/RNG/scheduler 상태를 저장하지 않으므로 **중간 step부터 정확히 resume할 수 없습니다.** 미완료 폴더를 보관한 뒤 해당 run을 처음부터 실행하세요.
- 50 step마다 저장하는 `adapter`만으로 학습을 이어 같은 실험으로 취급하지 마세요.
- OOM 시 다른 프로세스를 종료하지 말고 GPU 여유를 확보하세요. 설정을 바꾸면 전체 비교군을 같은 설정으로 다시 돌려야 합니다.

```bash
# 예: vanilla seed1이 미완료일 때만, 경로를 확인하고 실행
mkdir -p outputs/interrupted
mv outputs/controls/vanilla_s1_n100 outputs/interrupted/vanilla_s1_n100_attempt1
python -u scripts/run_suite.py --gpu 0 --seeds 1 --output outputs/controls
```

## 7. 완료 확인, 통계, LaTeX 표

여러 서버 결과 폴더를 한 `outputs/controls/` 아래 모으세요. 같은 이름의 디렉터리가 두 개 있으면 먼저 config/source hash를 비교하고 자동 덮어쓰지 마세요. `result.json`, `config.json`, `train.jsonl`, `eval_*.json`, source snapshots를 함께 보존합니다.

```bash
python rebuttal_3213/summarize_controls.py --root outputs/controls \
  --arms vanilla,split,cgh_strict,pivot_rsft,cgh_strict_nois --seeds 1,2,3
python scripts/export_table.py outputs/controls/summary.json --output outputs/controls_table.tex
```

한 run이라도 빠지면 summary는 실패합니다. 초기화, trainer/계수 hash, 패키지 버전, train/eval ID, 주요 설정을 검사합니다. 같은 환경에서도 GPU/커널 차이로 bitwise 동일함은 보장되지 않습니다. semantic ID와 seed 두 축을 resample한 exploratory CI를 계산합니다. **150 language-item은 독립적인 150문제가 아니라 50 semantic IDs × 3 languages입니다.**

LaTeX 문서에서 `\input{controls_table.tex}`로 표를 넣습니다. 원래 논문에서 사용하는 표 자동화와 통합하려면 회수한 complete runs를 원래 `rebuttal_3213/runs/` 아래 검토해 가져온 뒤 원래 `export_latex.py`를 실행하세요. 이 저장소에는 미완료 성능을 채운 표가 없습니다.

```bash
# adapter를 제외한 작은 결과 묶음. adapter는 서버에 별도 보관하세요.
tar --exclude='*/adapter' -czf gate-results.tar.gz outputs/controls
# 받을 서버에서 압축을 별도 staging 폴더에 풀고 완성도/충돌 검사 후 통합
```

## 8. 해석과 반박 제출

모든 방법은 prompt마다 원래 4개 + 추가 2개 응답을 생성하지만 **token/FLOP 예산이 같지는 않습니다.** token·wall time을 기록합니다. 새 engine은 LoRA 초기화 순서, dropout=0, full-softmax sampling, EOS 보존 등이 과거 실행과 다릅니다. 새 CGH 결과를 옛 vanilla 숫자와 직접 비교하지 마세요. clipped/root ratio와 mean-token score를 쓰는 실제 업데이트에 exact sequence-level unbiasedness를 주장하지 마세요.

별도 real-model diagnostic은 `train_controls.py --diagnostic-prompts 12 --diagnostic-draws 16`으로 실행할 수 있지만 원래 서버에서 이미 진행 중이므로 기본 원격 작업에서 제외했습니다. 단일 LoRA block의 유한 표본 비교이며 true/full gradient 측정이 아닙니다. 낮은 ESS에서는 cosine 방향 비교도 불안정합니다.

이번 author response는 **reviewer마다 Author response textbox에 2,000자 이하 댓글 1개**입니다. Official Comment에 넣지 않습니다. 코드·LaTeX 파일은 내부 재현/개정용이며, 이 GitHub 링크를 반박 본문에 넣어 읽도록 요구하지 마세요. 완성된 추가 결과 중 직접 관련된 것만 숫자와 조건을 짧게 적습니다. smoke, 미완료 run, 선택적으로 좋은 seed만 고른 결과는 넣지 않습니다.
