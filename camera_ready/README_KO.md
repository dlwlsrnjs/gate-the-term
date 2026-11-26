# Camera-ready 추가실험: 설치, 데이터, 서버 인계, 결과 취합

이 문서가 2026-11 추가실험의 기준입니다. 저장소의 이전 `HANDOFF.md`와 50문제 평가 명령은 이전 rebuttal 기록입니다. 이번에는 **190개 학습 / 60개 MGSM holdout**, 핵심 6조건 × 3 seeds, 새 원문 2,069개에 대한 외부 평가를 사용합니다. 이전 논문의 숫자와 새 실행을 섞어 비교하지 않습니다.

현재 원래 서버에서는 전체 큐가 실행 중입니다. [SOURCE_STATUS.json](SOURCE_STATUS.json)은 공개 시점의 스냅샷이며 실시간 상태가 아닙니다. **같은 실험을 다른 서버에 배정하려면 4절의 인계를 먼저 실행합니다.** 별도의 전체 재현을 할 경우에는 새로운 output root를 사용합니다.

## 1. 필요한 환경

- 검증 환경: Linux x86-64, Python 3.11, H100 80GB, PyTorch 2.5.1+cu121.
- BF16을 지원하는 GPU가 필요합니다. 실행 전 24,000 MiB 여유를 검사합니다. 다른 GPU에서의 메모리/속도는 별도로 확인합니다.
- 모델과 환경, 체크포인트를 위해 디스크 여유 50GB 이상을 권장합니다. Qwen 가중치는 GitHub에 넣지 않고 동일 revision으로 받습니다.
- 정확한 패키지 버전: [environment_source.json](environment_source.json), [requirements-camera-ready.txt](../requirements-camera-ready.txt).
- GPU별로 실행기 하나를 사용합니다. `--gpu`는 `nvidia-smi`의 물리 GPU 번호입니다. 별도로 `CUDA_VISIBLE_DEVICES`를 설정하지 않습니다.

```bash
git clone https://github.com/dlwlsrnjs/gate-the-term.git
cd gate-the-term
# 다른 서버도 같은 commit을 사용하도록 이 값을 기록합니다.
git rev-parse HEAD
bash scripts/setup_camera_ready.sh
source .venv/bin/activate
python camera_ready/prepare.py --model
python camera_ready/verify.py --gpu 0
```

Python 실행 파일이 다르면 `PYTHON_BIN=/path/to/python3.11 bash scripts/setup_camera_ready.sh`를 사용합니다. 설치 실패 시 임의로 버전을 바꾸지 말고 오류와 변경 환경을 기록합니다. `setup_camera_ready.sh`는 저장소 내부 `.venv`에 설치하며, 기존 시스템 환경은 바꾸지 않습니다.

모델은 `Qwen/Qwen2.5-7B-Instruct`의 **`a09a35458c702b33eeacc393d103063234e8bc28`** revision입니다. 준비된 절대경로는 각 서버의 `scripts/model_path.txt`에 기록됩니다. 이 파일은 서버끼리 복사하지 않습니다. 이미 같은 snapshot이 있으면 다음 명령으로 등록할 수 있습니다. 디렉터리의 마지막 이름은 revision을 유지합니다.

```bash
python camera_ready/prepare.py --model-directory /path/to/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
```

오프라인 서버에는 인터넷이 되는 서버에서 준비한 snapshot을 **심볼릭 링크가 아닌 실제 파일로** 복사합니다. Hugging Face cache snapshot에는 링크가 있으므로 `rsync -aL` 등을 사용합니다. 가중치/adapter, 가상환경, 토큰·자격증명은 Git에 올리지 않습니다.

## 2. 포함된 데이터와 재구성

설치 스크립트는 저장소에 포함된 압축 데이터를 풀고 SHA256을 검증합니다. 데이터 준비만 다시 하려면 다음 명령을 사용합니다.

```bash
python camera_ready/prepare.py --data-only
python camera_ready/verify.py --cpu-only
```

| 평가셋 | 원문 문제 수 | 언어 | 평가 입력 수 |
|---|---:|---|---:|
| MGSM holdout | 60 | bn, te, th | 180 |
| Indic-GSM8K 추가 평가 | 1,069 | bn, te | 2,138 |
| MSVAMP 추가 평가 | 1,000 | bn, th | 2,000 |

외부 평가에는 원문이 다른 **2,069개 문제 / 4,138개 언어별 입력**이 있습니다. Indic-GSM8K 1,319개 중 기존 MGSM 250개를 영어 원문 정규화 매칭으로 모두 제외했습니다. MSVAMP와도 같은 정규화 기준의 중복은 없습니다. 근접한 변형 문제나 사전학습 오염까지 없다는 뜻은 아닙니다.

MSVAMP의 실제 공개 파일에는 Telugu가 없습니다. Telugu 결과를 임의로 만들어 세 언어 평균으로 보고하지 않습니다. 외부 자료의 번역 출처는 원래의 수작업 MGSM과 다르므로 데이터셋별·언어별 결과를 분리합니다. 평가 입력에는 target-language 질문만 넣으며 영어 원문과 정답은 모델에 보여주지 않습니다.

파일과 체크섬은 [data/assets_manifest.json](data/assets_manifest.json), 데이터 출처와 고정 revision은 [DATA_SOURCES.md](DATA_SOURCES.md)에 있습니다. 원자료부터 재구성할 때는:

```bash
python camera_ready/prepare.py --download-raw
python camera_ready/rebuild_evaluation.py
python camera_ready/verify.py --cpu-only
```

## 3. 먼저 동작 확인

```bash
python scripts/run_suite.py --gpu 0 --arms cgh_strict,component_zvp \
  --seeds 3214 --steps 1 --eval-n 1 --output outputs/camera_ready_smoke
python camera_ready/evaluate_expanded.py --gpu 0 --limit-items 2 \
  --batch-size 2 --output outputs/expanded_smoke
```

smoke 결과는 논문 성능 표에 포함하지 않습니다. 원래 서버에서 학습 coefficient 검사, 실제 모델의 작은 외부 평가, batch 16 처리 검사를 통과했습니다. 새 서버에서도 먼저 확인합니다.

## 4. 실행 중인 원래 서버에서 다른 서버로 인계

이 단계는 **원래 큐를 실행 중인 서버에서** 합니다. 다른 서버에서는 `/proc`의 PID를 확인할 수 없으므로 인계를 대신 실행할 수 없습니다.

```bash
# 먼저 상태만 확인: 작업을 멈추지 않습니다.
python camera_ready/source_handoff.py \
  --source-root /path/to/paper_eacl/camera_ready_experiments

# 인계할 때: 큐 관리자만 일시 정지하고 할당 파일 생성
python camera_ready/source_handoff.py \
  --source-root /path/to/paper_eacl/camera_ready_experiments \
  --pause-source --output outputs/source_assignment.json
```

스크립트는 PID의 실제 명령을 확인한 후 학습/평가 **관리자만** 정지합니다. 현재 진행 중인 학습 자식은 100 steps를 끝내도록 두고, 그 작업은 원래 서버에 예약합니다. 나머지 작업만 `remote_jobs`에 넣습니다. 생성된 `source_assignment.json`을 새 서버에 복사합니다. 당시 상태에 따라 예약·완료·원격 작업 수는 달라집니다.

다른 서버가 원격 작업을 맡은 동안 예전 관리자를 재개하지 않습니다. 원격 결과를 취합한 뒤에는 예전 관리자 대신 이 저장소의 실행기를 사용합니다. 진행 중이던 원래 두 작업도 완료 후 결과 폴더와 adapter를 함께 회수합니다. 공개된 `SOURCE_STATUS.json`만 보고 예전 배정표를 재사용하지 않습니다.

## 5. 핵심 실험 실행

| arm | 역할 |
|---|---|
| `vanilla` | 동일 환경 scalar GRPO |
| `split` | target/auxiliary 정규화만 분리 |
| `cgh_strict` | 원래 auxiliary 계수를 유지하고 all-wrong target만 교체 |
| `pivot_rsft` | 영어 pivot으로 얻은 정답 응답을 원래 질문 아래 직접 학습 |
| `cgh_legacy` | 원래 CGH coefficient 분기를 새 공통 engine에서 실행 |
| `cgh_strict_nois` | strict CGH에서 importance weight를 1로 설정 |

각 조건은 seeds 1,2,3, 100 steps입니다. 공통 engine은 LoRA 생성 전에 seed를 설정하고 dropout을 0으로 고정하며, prompt마다 원래 응답 4개와 추가 응답 2개를 생성합니다. 이는 응답 수 통제이며 token/FLOP 통제는 아닙니다.

```bash
# 인계 받은 작업만 한 GPU에서 순서대로 실행
python -u camera_ready/run.py --gpu 0 --assignment source_assignment.json \
  --output outputs/camera_ready > primary.log 2>&1
```

두 GPU가 있으면 별도 tmux/터미널에서 다음 두 명령을 실행합니다. 같은 arm/seed가 양쪽에 배정되지 않습니다.

```bash
# GPU 0
python -u camera_ready/run.py --gpu 0 --assignment source_assignment.json \
  --arms vanilla,cgh_strict,cgh_legacy --output outputs/camera_ready > gpu0.log 2>&1
# GPU 1
python -u camera_ready/run.py --gpu 1 --assignment source_assignment.json \
  --arms split,pivot_rsft,cgh_strict_nois --output outputs/camera_ready > gpu1.log 2>&1
```

여러 서버를 쓰면 같은 commit·환경을 준비하고 `--seeds 1`, `--seeds 2`, `--seeds 3`으로 나눌 수도 있습니다. 서로 다른 서버 사이에는 자동 잠금이 없으므로 한 arm/seed는 한 서버에만 배정합니다. 새로 전체를 재현하려면 `--assignment`를 생략하고 별도의 output root에서 실행합니다. 명령만 확인할 때는 `--dry-run`을 붙입니다.

후순위는 `whole_gate`, `q_resample`, `component_zvp`입니다. 핵심 18회를 먼저 완성한 뒤 실행합니다.

```bash
python -u camera_ready/run.py --gpu 0 --plan secondary \
  --assignment source_assignment.json --output outputs/camera_ready > secondary.log 2>&1
```

`component_zvp`는 component trigger를 사용하는 token-entropy adaptation입니다. `q_resample`도 full DAPO가 아닙니다. 이 비교를 원 논문 전체 시스템의 재현으로 부르지 않습니다.

## 6. 상태 확인, 실패 후 재실행

```bash
tail -f primary.log
cat outputs/camera_ready/status_gpu0.json
tail -f outputs/camera_ready/cgh_strict_s1_n100.log
nvidia-smi
```

완료된 `result.json`과 config가 일치하면 해당 작업은 건너뜁니다. 다른 서버로 옮겨도 고정 revision의 모델 경로 차이는 허용합니다. 데이터·engine·평가 조건 차이는 허용하지 않습니다.

미완료 디렉터리가 있으면 실행기는 중단합니다. **Adam/RNG 상태를 저장하지 않으므로 중간 step부터 정확히 resume할 수 없습니다.** 미완료 폴더를 `outputs/interrupted/` 등으로 보관한 뒤 그 작업을 처음부터 시작합니다. 50-step adapter만 이어 학습한 결과를 동일한 100-step run으로 취급하지 않습니다. 자동 재시도나 좋은 seed만 선택하는 처리는 없습니다.

## 7. 여러 서버 결과 취합

각 서버의 결과를 먼저 서로 다른 staging 폴더로 가져옵니다. `config.json`, `result.json`, `train.jsonl`, `eval_*.json`, source snapshots를 보존합니다. 외부 평가를 하려면 **adapter도 필요**합니다.

```bash
# 각 서버에서 전체 결과 묶음: Git에는 올리지 않습니다.
tar -czf camera-ready-results.tar.gz outputs/camera_ready
# 수신 서버에서 A/B 각각 다른 staging 디렉터리에 압축을 풉니다.
python camera_ready/merge_results.py \
  --roots stagingA/outputs/camera_ready stagingB/outputs/camera_ready \
  --destination outputs/merged --with-adapters
```

중복 job의 파일이 다르면 덮어쓰지 않고 실패합니다. 미완료 run은 복사하지 않고 위치를 출력합니다. 최종 집계는 요청된 모든 arm/seed가 완료돼야 가능합니다.

## 8. 확장 평가

핵심 체크포인트를 모은 뒤 실행합니다. 모델마다 4,138개 입력을 평가하며, base는 전체 서버 중 한 번만 실행합니다.

```bash
# GPU 0: 세 조건과 base
python -u camera_ready/evaluate.py --gpu 0 --train-root outputs/merged \
  --arms vanilla,cgh_strict,cgh_legacy --include-base \
  --output outputs/expanded > expanded_gpu0.log 2>&1
# GPU 1: 나머지 세 조건
python -u camera_ready/evaluate.py --gpu 1 --train-root outputs/merged \
  --arms split,pivot_rsft,cgh_strict_nois \
  --output outputs/expanded > expanded_gpu1.log 2>&1
```

여러 서버에서는 평가도 arm/seed로 나눕니다. 결과는 같은 tag끼리 충돌하지 않게 한 `outputs/expanded/`에 모읍니다. 기본 설정은 greedy, 320 new tokens, batch 16이며 코드/데이터/adapter 체크섬을 기록합니다. source snapshot의 이전 절대경로 manifest를 쓰는 평가와 이 portable 평가를 임의로 섞지 않습니다.

H100에서 batch 16으로 64개 출력을 생성한 측정은 약 50초였습니다. 단순 환산으로 모델당 약 55분이며, 실제 전체 데이터의 길이·장비·다른 작업에 따라 달라집니다. 원래 일정은 두 GPU 기준 전체 약 30–45시간입니다. 다른 서버를 추가하면 겹치지 않는 작업을 나누어 줄일 수 있습니다.

## 9. 통계와 LaTeX 표

```bash
python rebuttal_3213/summarize_controls.py --root outputs/merged \
  --arms vanilla,split,cgh_strict,pivot_rsft,cgh_legacy,cgh_strict_nois --seeds 1,2,3
python scripts/export_table.py outputs/merged/summary.json \
  --output outputs/controls_table.tex
python camera_ready/summarize_expanded.py --root outputs/expanded
```

MGSM은 독립적인 원문 60개를 세 언어로 번역한 것이고, 외부 평가는 데이터셋별 원문 1,069개/1,000개입니다. 번역과 seed 반복을 새 독립 문제로 세지 않습니다. seed와 원문 ID를 함께 resample하는 탐색적 crossed bootstrap을 사용하고, 데이터셋·언어별 정확도와 language consistency를 함께 보고합니다. 초기 LoRA hash, engine/계수 hash, 버전, split을 검사하며 빠진 seed를 자동으로 빼지 않습니다.

게재본 반영 방향과 해석 제한은 [PAPER_INTEGRATION.md](PAPER_INTEGRATION.md)에 있습니다. 완료되지 않은 성능 수치는 이 저장소에 넣지 않았습니다.
