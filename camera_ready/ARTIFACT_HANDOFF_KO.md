# 최종 후속 실험 인계: 18개 adapter와 전체 결과 받기

이 안내는 2026-11-26에 완료한 핵심 6조건 × 3 seeds의 결과를 다른 서버로 옮겨, 후순위 학습 9회와 외부 평가 19개 모델을 실행하기 위한 것입니다. 코드·데이터 준비 방법은 GitHub, 대용량 adapter·원본 결과는 Hugging Face에서 받습니다. 고정 HF commit과 manifest SHA256은 [artifacts_release.json](artifacts_release.json)에 기록합니다.

- GitHub: https://github.com/dlwlsrnjs/gate-the-term
- Hugging Face: https://huggingface.co/jin-kwon/gate-the-term-camera-ready-20261126
- Base: `Qwen/Qwen2.5-7B-Instruct`, revision `a09a35458c702b33eeacc393d103063234e8bc28`

## 포함한 것

18개 최종 LoRA safetensors(합계 약 2.9 GB), 원본 학습 config·200개 prompt-level 기록/run·stdout 로그·최종 언어별 응답·result.json·trainer/coefficient snapshot을 제공합니다. 압축 MGSM/Indic-GSM8K/MSVAMP 입력, split·데이터 및 코드 SHA256, 환경 버전, 분석 결과와 실행 가능한 코드 archive도 포함합니다.

업로드 사본의 adapter config만 Base 모델의 공식 ID·revision으로 바꿨습니다. 원본 adapter config·README도 별도로 남겼으며, 가중치와 원래 로컬 결과 파일은 변경하지 않았습니다. 원본 config에 남은 절대경로는 출처 기록이며 다른 서버의 실행 경로로 사용하지 않습니다.

Base 7B 가중치는 공식 저장소에서 같은 revision으로 받습니다. optimizer/RNG state는 저장하지 않았습니다. 중간 학습을 정확히 resume하는 방식은 지원하지 않으며, 남은 세 방법은 Base에서 각각 새로 학습합니다.

## 1. 환경과 코드

권장 검증 환경은 Linux, Python 3.11, H100 80GB, torch 2.5.1+cu121입니다. 같은 비교에서는 고정 패키지·모델·데이터·G·응답 길이·seed를 유지합니다.

```bash
git clone https://github.com/dlwlsrnjs/gate-the-term.git
cd gate-the-term
bash scripts/setup_camera_ready.sh
source .venv/bin/activate
```

## 2. 가중치·결과 다운로드와 검증

공개 저장소이므로 다운로드에는 토큰이 필요하지 않습니다. 다음 명령은 release JSON에 기록된 immutable HF commit을 받고 모든 파일의 SHA256·크기와 18회 완료 결과를 검사합니다.

```bash
python camera_ready/artifacts.py download --directory received/primary
python camera_ready/prepare.py --model
python camera_ready/verify.py --gpu 0
python camera_ready/verify.py --gpu 1
```

복구용 코드 archive는 `received/primary/code/gate-the-term-source.tar.gz`에 있습니다. 그 archive는 manifest의 `code_base_commit`에 해당하며 학습·평가 engine과 데이터가 고정돼 있습니다. 최신 다운로드 도구와 release receipt는 현재 GitHub를 사용합니다.

GPU가 한 장이면 GPU 1 검증·실행을 생략하고 GPU 0에서 순서대로 실행합니다. 별도로 `CUDA_VISIBLE_DEVICES`를 설정하지 않습니다. GPU 0/1은 물리 GPU 번호입니다.

## 3. 확장 평가: Base 1개 + adapter 18개

원문 2,069개/언어 입력 4,138개를 모델마다 평가합니다. Indic-GSM8K는 bn/te 1,069개씩, MSVAMP는 bn/th 1,000개씩이며 MSVAMP Telugu는 없습니다. 평가 입력에 영어 reference·정답은 넣지 않습니다. Greedy, 최대 320 new tokens, batch 16을 유지합니다.

원격에서 이미 같은 tag를 실행 중인지 먼저 분담을 확인합니다. 두 터미널 또는 tmux 창에서 아래를 실행합니다. 같은 GPU에서 학습·평가 관리자를 겹쳐 실행하지 않습니다.

```bash
# GPU 0: 세 조건 × 3 seeds + Base = 10개
python -u camera_ready/evaluate.py --gpu 0 \
  --train-root received/primary/runs \
  --arms vanilla,cgh_strict,cgh_legacy --include-base \
  --output outputs/final_expanded > final_expanded_gpu0.log 2>&1
```

```bash
# GPU 1: 나머지 세 조건 × 3 seeds = 9개
python -u camera_ready/evaluate.py --gpu 1 \
  --train-root received/primary/runs \
  --arms split,pivot_rsft,cgh_strict_nois \
  --output outputs/final_expanded > final_expanded_gpu1.log 2>&1
```

단일 GPU는 `--arms`를 생략하고 `--include-base`를 넣어 19개를 모두 실행합니다. Base는 전체 환경에서 한 번만 평가합니다. 다른 서버와 tag를 나눌 때는 `--arms`와 `--seeds`를 명시합니다.

완료된 동일 설정의 tag는 재실행 시 건너뜁니다. 미완료 tag 디렉터리가 남으면 자동 덮어쓰기하지 않으므로 해당 폴더를 별도 보관하고 그 tag를 처음부터 다시 실행합니다. 원본 prediction과 config를 보존합니다.

19개가 모두 완료된 뒤:

```bash
python camera_ready/summarize_expanded.py --root outputs/final_expanded
```

## 4. 후순위 학습 9회

`whole_gate`, `q_resample`, `component_zvp` × seeds 1,2,3을 각각 Base에서 100 steps 학습하고 MGSM holdout 60개 × 3언어를 평가합니다. 받은 CGH adapter를 이어 학습하지 않습니다. 실행 전 다른 서버의 동일 arm/seed와 중복되지 않게 배정합니다.

평가와 동일 GPU를 쓴다면 평가 종료 후 실행합니다. `--role remote`에는 과거 로컬의 2026-11-26 22시 cutoff가 적용되지 않습니다.

```bash
python -u camera_ready/run_parallel.py --gpus 0,1 --workers-per-gpu 4 \
  --allocation camera_ready/allocation_2026-11-26.json --role remote \
  --output outputs/remote_controls --state-dir outputs/remote_state \
  > remote_parallel.log 2>&1
```

Entropy 조건의 메모리 상한은 실행기가 별도로 제한합니다. GPU 한 장이면 `--gpus 0`을 씁니다. 실패·미완료 run은 원본을 보관하고 seed 초기화부터 다시 실행하며 optimizer/RNG resume로 취급하지 않습니다.

## 5. 결과 취합

```bash
python camera_ready/merge_results.py \
  --roots received/primary/runs outputs/remote_controls \
  --destination outputs/merged --with-adapters
python rebuttal_3213/summarize_controls.py --root outputs/merged \
  --arms vanilla,split,cgh_strict,pivot_rsft,cgh_legacy,cgh_strict_nois,whole_gate,q_resample,component_zvp \
  --seeds 1,2,3
```

확장 평가 전체와 후순위 학습 9회는 이번 업로드에 완료 결과가 들어 있다는 뜻이 아닙니다. 새 결과를 받은 뒤 benchmark별·언어별·seed별로 취합합니다. 원문 60개/2,069개를 번역·seed 반복으로 늘려 세지 않습니다. 외부 test set으로 모델이나 seed를 고르지 않습니다.

## 6. 그대로 전달할 인계 프롬프트

아래 내용을 다른 서버의 작업에 붙여 넣습니다.

> GitHub `https://github.com/dlwlsrnjs/gate-the-term`의 `camera_ready/ARTIFACT_HANDOFF_KO.md`를 따라 최종 후속 실험을 진행해 주세요. Hugging Face `jin-kwon/gate-the-term-camera-ready-20261126`의 고정 commit `63cd7187e29bb27da0bcf238884ed0923c0682cc`에 핵심 18회 adapter·원본 결과·데이터·환경이 있습니다. `camera_ready/artifacts.py download --directory received/primary`로 받고 전체 SHA256 검증을 통과시켜 주세요. Base는 `Qwen/Qwen2.5-7B-Instruct` revision `a09a35458c702b33eeacc393d103063234e8bc28`를 사용합니다. 완료된 핵심 18회 학습은 다시 돌리지 않습니다. 남은 작업은 (1) Base + 핵심 18개 adapter의 외부 평가 19개 모델, 모델당 2,069개 원문/4,138개 입력, (2) whole_gate/q_resample/component_zvp × seeds 1,2,3의 새 학습 9회입니다. 같은 GPU에서는 두 단계를 겹치지 말고 기존 다른 서버 실행과 tag가 중복되는지 먼저 확인해 주세요. 환경·평가 batch 16·greedy·최대 320 tokens·100 training steps·190/60 MGSM 분할을 유지합니다. Smoke나 미완료 결과를 연구 성능으로 섞지 말고, 완료 후 per-seed/per-language 결과와 paired uncertainty 및 원본 파일을 보존해 Markdown과 GitHub에 업데이트해 주세요.
