"""Export an explicitly provisional, auditable snapshot of complete fresh runs.

Incomplete runs are status only. Per-seed intervals resample semantic problem IDs,
keeping the three translations together; they do not cover training-seed variance.
"""
import argparse
import csv
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import heapq
import io
import json
from pathlib import Path
import statistics
import time
import numpy as np

HERE = Path(__file__).resolve().parent
LANGS = ["bn", "te", "th"]
FIXED = ["source_sha256", "coefficients_sha256", "torch", "transformers", "peft",
         "train_ids", "eval_ids", "dropout", "max_new", "extra", "eval_batch", "entropy_alpha"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "results/latest")
    args = parser.parse_args()
    protocol = json.loads((HERE / "protocol.json").read_text())
    now = time.time()
    tz = ZoneInfo("Asia/Seoul")
    timestamp = datetime.fromtimestamp(now, tz).isoformat(timespec="seconds")
    fmt = lambda value: datetime.fromtimestamp(value, tz).strftime("%H:%M")
    runs, observations, provenance = {}, [], []
    for tag in protocol["jobs"]:
        directory = args.root / tag
        path = directory / "result.json"
        if not path.exists():
            continue
        cfg = json.loads((directory / "config.json").read_text())
        result = json.loads(path.read_text())
        assert cfg["eval_n"] == 60 and cfg["steps"] == result["steps"] == 100
        assert cfg["source_sha256"] == digest(HERE.parent / "rebuttal_3213/train_controls.py")
        assert cfg["coefficients_sha256"] == digest(HERE.parent / "rebuttal_3213/control_coefficients.py")
        assert Path(cfg["model"]).name == protocol["model_revision"]
        assert cfg["dropout"] == 0 and cfg["max_new"] == 320 and cfg["extra"] == 2 and cfg["eval_batch"] == 8
        assert tag == f"{result['arm']}_s{result['seed']}_n100"
        assert not set(cfg["train_ids"]) & set(cfg["eval_ids"])
        rows = {(lang, row["i"]): row for lang, data in result["eval_end"].items()
                for row in data["per_problem"]}
        assert set(result["eval_end"]) == set(LANGS) and len(rows) == 180
        assert all({item for lang, item in rows if lang == lg} == set(cfg["eval_ids"]) for lg in LANGS)
        assert all(row["corr"] == row["corr_raw"] * row["lang"] for row in rows.values())
        assert all(row[key] in [0., 1.] for row in rows.values() for key in ["corr", "corr_raw", "lang"])
        history = [json.loads(line) for line in (directory / "train.jsonl").read_text().splitlines()]
        assert len(history) == 200 and len({(row["step"], row["batch"]) for row in history}) == 200
        assert all(row["generated_responses"] == 6 for row in history)
        assert result["total_generated_tokens"] == sum(row["generated_tokens"] for row in history)
        def counts(selected):
            return {"n": len(selected), **{key: int(sum(row[key] for row in selected))
                    for key in ["corr", "corr_raw", "lang"]}}
        metrics = {"tag": tag, "arm": result["arm"], "seed": result["seed"],
                   "overall": counts(list(rows.values())),
                   "by_language": {lg: counts([row for (lang, item), row in rows.items() if lang == lg]) for lg in LANGS},
                   "training_groups": {"n": len(history), "target_degenerate": sum(row["degenerate"] for row in history),
                                       "all_wrong": sum(row["all_wrong"] for row in history),
                                       "scalar_zero": sum(row["scalar_zero"] for row in history),
                                       "admitted_extras": sum(row["admitted"] for row in history),
                                       "generated_positive_extras": int(sum(sum(row["extra_target"]) for row in history)),
                                       "admission_type": "positive-only" if result["arm"] in ["cgh_strict", "cgh_strict_nois", "pivot_rsft"] else "contrast" if result["arm"] == "cgh_legacy" else "none"},
                   "generated_tokens": result["total_generated_tokens"],
                   "prompt_tokens": result["total_prompt_tokens"], "elapsed_s": result["elapsed_s"]}
        runs[result["arm"], result["seed"]] = {"cfg": cfg, "rows": rows, "history": history, "metrics": metrics}
        provenance.append({"tag": tag, "result_sha256": digest(path), "config_sha256": digest(directory / "config.json"),
                           "train_log_sha256": digest(directory / "train.jsonl"),
                           "initial_lora_sha256": cfg["initial_lora_sha256"],
                           "engine_sha256": cfg["source_sha256"], "coefficients_sha256": cfg["coefficients_sha256"],
                           "model_revision": protocol["model_revision"],
                           "versions": {key: cfg[key] for key in ["torch", "transformers", "peft"]},
                           "eval_ids": cfg["eval_ids"], "training_steps": 100})
        for (lg, item), row in sorted(rows.items()):
            observations.append({"arm": result["arm"], "seed": result["seed"], "language": lg,
                                 "semantic_id": item, **{key: row[key] for key in ["corr", "corr_raw", "lang"]}})
    arm_order = protocol["primary"] + protocol["secondary"]
    pairs = [("vanilla", arm) for arm in arm_order if arm != "vanilla"] + [
        ("split", "cgh_strict"), ("split", "pivot_rsft"),
        ("pivot_rsft", "cgh_strict"), ("pivot_rsft", "cgh_strict_nois"),
        ("cgh_strict_nois", "cgh_strict"), ("cgh_legacy", "cgh_strict"),
        ("q_resample", "cgh_strict"), ("whole_gate", "cgh_strict"), ("component_zvp", "cgh_strict")]
    comparisons = []
    for seed, (left, arm) in [(seed, pair) for seed in protocol["seeds"] for pair in pairs]:
        if (arm, seed) not in runs or (left, seed) not in runs:
            continue
        run = runs[arm, seed]
        baseline = runs[left, seed]
        assert all(run["cfg"][key] == baseline["cfg"][key] for key in FIXED)
        assert run["cfg"]["initial_lora_sha256"] == baseline["cfg"]["initial_lora_sha256"]
        assert all([(a["step"], a["batch"], a["lang"], a["i"]) == (b["step"], b["batch"], b["lang"], b["i"])
                    for a, b in zip(run["history"], baseline["history"])])
        ids = sorted(run["cfg"]["eval_ids"])
        delta = np.array([np.mean([run["rows"][lg, item]["corr"] - baseline["rows"][lg, item]["corr"]
                                  for lg in LANGS]) for item in ids])
        rng = np.random.default_rng(3213)
        bootstrap = delta[rng.integers(len(ids), size=(20000, len(ids)))].mean(axis=1)
        paired = {}
        for lg in LANGS:
            fixes = [item for item in ids if baseline["rows"][lg, item]["corr"] == 0 and run["rows"][lg, item]["corr"] == 1]
            breaks = [item for item in ids if baseline["rows"][lg, item]["corr"] == 1 and run["rows"][lg, item]["corr"] == 0]
            paired[lg] = {"fixed_ids": fixes, "broken_ids": breaks, "delta_pp": (len(fixes) - len(breaks)) / 60 * 100}
        delta_pp = (run["metrics"]["overall"]["corr"] - baseline["metrics"]["overall"]["corr"]) / 180 * 100
        comparisons.append({"baseline": left, "arm": arm, "seed": seed, "delta_pp": delta_pp,
                            "conditional_semantic_cluster_95ci_pp": (np.quantile(bootstrap, [.025, .975]) * 100).tolist(),
                            "bootstrap_replicates": 20000, "bootstrap_rng_seed": 3213,
                            "scope": "Exploratory fixed-seed/model pair; selected languages fixed; training-seed uncertainty NOT included; intervals not multiplicity-adjusted.",
                            "paired_by_language": paired})
    state = json.loads(args.state.read_text())
    workers, rates, overhead = [], [], []
    for tag, record in state["workers"].items():
        log = args.state.parent / (tag + ".log")
        step_rows = []
        evaluated_languages = []
        if log.exists():
            for line in log.read_text(errors="replace").splitlines():
                if line.startswith('{"step"'):
                    step_rows.append(json.loads(line))
                elif line.startswith("evaluated "):
                    parts = line.split()
                    if len(parts) == 3 and parts[1] in LANGS and parts[2] == "60":
                        evaluated_languages.append(parts[1])
        last = step_rows[-1] if step_rows else None
        rate = ((last["elapsed_min"] - step_rows[-11]["elapsed_min"]) / 10) if len(step_rows) >= 11 else None
        if rate is not None:
            rates.append(rate)
        workers.append({"job": tag, "gpu": record["gpu"], "step": last["step"] if last else 0,
                        "phase": "final_evaluation" if last and last["step"] == 100 else "training",
                        "evaluated_languages": sorted(set(evaluated_languages)),
                        "recent_minutes_per_step": rate, "last_log_epoch": log.stat().st_mtime if log.exists() else now})
    for run in runs.values():
        tag = run["metrics"]["tag"]
        log = args.state.parent / (tag + ".log")
        values = [json.loads(line) for line in log.read_text(errors="replace").splitlines() if line.startswith('{"step"')]
        if values:
            overhead.append(run["metrics"]["elapsed_s"] / 60 - values[-1]["elapsed_min"])
    forecasts = []
    if rates and workers:
        rate = statistics.median(rates)
        post = statistics.median(overhead) if overhead else 15.
        deadline = datetime.fromisoformat(state["deadline"]).timestamp()
        heap = []
        for worker in workers:
            own = worker["recent_minutes_per_step"] or rate
            finish = max(now, worker["last_log_epoch"] + (100 - worker["step"]) * own * 60) + post * 60
            heapq.heappush(heap, (finish, worker["gpu"]))
            forecasts.append({"job": worker["job"], "finish_kst": fmt(finish), "kind": "running"})
        for tag in state["pending"]:
            available, gpu = heapq.heappop(heap)
            start = available + 30
            admitted = start + 235 * 60 < deadline
            finish = start + (rate * 100 + post + 2) * 60
            forecasts.append({"job": tag, "start_kst": fmt(start), "finish_kst": fmt(finish),
                              "kind": "queued", "admitted_before_cutoff_estimate": admitted})
            heapq.heappush(heap, (finish, gpu))
    completed_metrics = [runs[key]["metrics"] for key in sorted(runs, key=lambda key: (key[1], arm_order.index(key[0])))]
    report = {"as_of_kst": timestamp, "scope": "PROVISIONAL completed fresh runs only; smoke and historical results excluded.",
              "coverage": "Only result files present in the supplied root. Uncollected other-host runs are unobserved, not failures.",
              "analysis_source_sha256": digest(Path(__file__)), "analysis_numpy_version": np.__version__,
              "planned_seeds": protocol["seeds"], "completed_runs": completed_metrics, "per_seed_comparisons": comparisons,
              "method_three_seed_means": {arm: None if any((arm, seed) not in runs for seed in protocol["seeds"]) else
                  float(np.mean([runs[arm, seed]["metrics"]["overall"]["corr"] / 180 for seed in protocol["seeds"]]))
                  for arm in protocol["primary"] + protocol["secondary"]},
              "sources": provenance}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "completed_results.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output / "completed_items.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["arm", "seed", "language", "semantic_id", "corr", "corr_raw", "lang"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(observations)
    progress = {"as_of_kst": timestamp, "local_state": state["state"],
                "local_completed_kst": datetime.fromisoformat(state["updated_utc"]).astimezone(tz).isoformat(timespec="seconds")
                    if state["state"] == "complete" else None,
                "local_deadline": state["deadline"], "completed_local": state["completed"],
                "active": workers, "pending_local": state["pending"], "failed": state["failed"],
                "remote_fallback_jobs": state["remote_fallback_jobs"], "gpu_resources": state["gpu_resources"],
                "forecast": forecasts, "remote_training_jobs": 9, "remote_external_evaluation_models": 19,
                "remote_execution_status_observed_here": False,
                "note": "Published snapshot, not live monitoring. Forecast uses recent step times and observed post-training evaluation duration."}
    (args.output / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
    lines = ["# 완료 실험 결과 — 잠정 분석", "", f"기준 시각: {timestamp}", "",
             "지정한 결과 폴더에서 확인한 fresh cohort의 완료 run만 포함합니다. 다른 서버 결과가 취합되지 않았다면 그 서버의 완료 여부는 미확인입니다. 단일 seed 관측은 방법의 최종 성능이나 세 seed 평균이 아닙니다.", "",
             "CGH와 pivot 직접학습의 상세 해석은 [직접 대조 분석](../CGH_PIVOT_KO.md)을 함께 확인합니다.", "",
             *(["핵심 여섯 조건의 세 seed가 모두 완료됐습니다. [3-seed 평균·언어별 결과·crossed 구간](THREE_SEED_RESULTS_KO.md)을 확인합니다.", ""]
               if all(report["method_three_seed_means"][arm] is not None for arm in protocol["primary"]) else []),
             "| 조건 | Seed | Bengali | Telugu | Thai | 전체 target 정확도 | Raw 정답 | Script 기준 통과 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m in completed_metrics:
        overall = m["overall"]
        language = [f"{m['by_language'][lg]['corr']}/60 ({m['by_language'][lg]['corr'] / 60:.2%})" for lg in LANGS]
        lines.append(f"| {m['arm']} | {m['seed']} | " + " | ".join(language) +
                     f" | {overall['corr']}/180 ({overall['corr'] / 180:.2%}) | {overall['corr_raw']}/180 | {overall['lang']}/180 |")
    lines += ["", "## 사전 계획한 세 seed 평균", "", "세 seed가 모두 완료된 방법만 평균을 표시합니다.", "",
              "| 조건 | 이 폴더에서 완료 확인한 seeds | 3-seed target 정확도 |", "|---|---|---:|"]
    for arm, mean in report["method_three_seed_means"].items():
        done = [str(seed) for seed in protocol["seeds"] if (arm, seed) in runs]
        lines.append(f"| {arm} | {', '.join(done) or '미완료 또는 미취합'} | " + (f"{mean:.2%}" if mean is not None else "미완료 — 보류") + " |")
    lines += ["", "Target 정확도는 수치 정답과 script 기준을 함께 요구합니다. Script 기준은 target script 비중 50% 이상인 결정적 규칙이며, 완전한 언어 식별이나 혼합 언어 부재를 보장하지 않습니다.",
              "", "## 기준선과 문항별 비교", ""]
    for comparison in comparisons:
        paired = comparison["paired_by_language"]
        fixed = sum(len(item["fixed_ids"]) for item in paired.values())
        broken = sum(len(item["broken_ids"]) for item in paired.values())
        low, high = comparison["conditional_semantic_cluster_95ci_pp"]
        left_label = "GRPO" if comparison["baseline"] == "vanilla" else comparison["baseline"]
        lines += [f"- Seed {comparison['seed']}, {comparison['arm']} − {left_label}: {comparison['delta_pp']:+.2f} pp; "
                  f"개선 {fixed}개 / 악화 {broken}개 언어–문항 결과. 원문 ID로 묶은 조건부 95% bootstrap 구간 [{low:+.2f}, {high:+.2f}] pp."]
    lines += ["", "이 탐색적 구간은 원문 60개를 resample하며 세 번역을 함께 유지합니다. 한 seed로 고정된 모델의 문항 변동만 반영하고 학습 seed 변동은 반영하지 않습니다. 다중 비교 보정은 하지 않았습니다. 전체 180개를 독립 문제로 취급하지 않습니다. 구간에 0이 포함된다는 이유로 두 방법의 동등성을 주장하지 않습니다.",
              "", "## 학습 중 성분 퇴화 진단", "",
              "| 조건 / seed | Target 퇴화 그룹 | All-wrong 그룹 | Scalar zero-variance 그룹 |",
              "|---|---:|---:|---:|"]
    for m in completed_metrics:
        t = m["training_groups"]
        lines.append(f"| {m['arm']} / {m['seed']} | {t['target_degenerate']}/{t['n']} | {t['all_wrong']}/{t['n']} | {t['scalar_zero']}/{t['n']} |")
    lines += ["", "## 추가 생성과 업데이트", "", "| 조건 / seed | 추가 생성 중 positive 수 / 400 | 채택된 extra 수 | 채택 규칙 | Generated tokens |",
              "|---|---:|---:|---|---:|"]
    for m in completed_metrics:
        t = m["training_groups"]
        lines.append(f"| {m['arm']} / {m['seed']} | {t['generated_positive_extras']}/400 | {t['admitted_extras']} | {t['admission_type']} | {m['generated_tokens']} |")
    lines += ["", "positive-only 조건의 채택 수는 positive extra 업데이트 수입니다. legacy의 contrast 채택은 같은 의미가 아니므로 직접 동치로 비교하지 않습니다. 같은 생성 개수라도 token/FLOP, 채택 수, gradient 크기는 같지 않습니다. 조건별 policy가 달라지므로 동일 conditioning을 써도 생성 응답은 동일하게 고정되지 않습니다."]
    lines += ["", "이는 각 run의 200개 학습 그룹에 대한 기술 통계입니다. 반복 문제와 변화하는 policy를 포함하므로 독립 표본의 효과 검정으로 읽지 않습니다.",
              "", "## 해석 범위", "",
              "성능과 영어 정보 효과의 비교는 [CGH·pivot 직접 대조](../CGH_PIVOT_KO.md), 각 리뷰 지적에 대한 판단은 [리뷰 해석](../REVIEWER_INTERPRETATION_KO.md)에 기록합니다. 이 문서는 완료된 수치를 자동 집계하며, 특정 한 seed의 해석을 이후 완료 결과에 대신 적용하지 않습니다.",
              "", "성분 퇴화 그룹과 scalar zero-variance 그룹의 빈도 차이는 진단 불일치에 대한 기술 통계입니다. 퇴화율 감소 자체를 repair 성공으로 해석하거나, 정확도 개선으로 population gradient의 복원·unbiasedness를 주장하지 않습니다.",
              "", "새 cohort는 초기화 순서, dropout, sampling과 평가 크기를 맞춰 다시 실행한 비교입니다. 기존 논문의 세 seed 수치와 직접 대조해 개선/악화를 판단하지 않습니다. 아직 세 seed가 끝나지 않은 방법 평균은 JSON에서 null로 남깁니다.",
              "", "## 재현과 출처", "",
              "문항별 수치는 completed_items.csv, paired ID·bootstrap·source SHA256은 completed_results.json에 있습니다. full response와 adapter는 원본 결과 폴더에 보존합니다.",
              "", "```bash", "python camera_ready/report_completed.py --root /path/to/camera_ready_experiments/runs \\",
              "  --state /path/to/camera_ready_experiments/status.json --output camera_ready/results/latest", "```", ""]
    (args.output / "RESULTS_KO.md").write_text("\n".join(lines))
    lines = ["# 실험 진행 현황", "", f"기준 시각: {timestamp}", "",
             f"로컬: 완료 {len(state['completed'])}/18, 실행 {len(workers)}, 대기 {len(state['pending'])}, 실패 {len(state['failed'])}.",
             f"로컬 관리자 상태: {state['state']}." + (f" 전체 완료 시각: {progress['local_completed_kst']}." if progress['local_completed_kst'] else ""),
             f"로컬 마감: {state['deadline']}. 다른 환경 배정은 후순위 학습 9회와 확장 평가 19개 모델입니다.",
             "다른 환경의 실제 실행/완료 상태는 이 서버에서 확인하지 않았습니다.",
             "", "## 실행 중", "", "| 작업 | GPU | Step / 100 | 단계 | 평가 완료 언어 |", "|---|---:|---:|---|---|"]
    lines += [f"| {w['job']} | {w['gpu']} | {w['step']} | "
              + ("최종 평가" if w['phase'] == 'final_evaluation' else "학습")
              + " | " + (", ".join(w['evaluated_languages']) or "—") + " |" for w in workers]
    if not workers:
        lines += ["", "없음."]
    lines += ["", "## 완료", ""] + [f"- {tag}" for tag in state["completed"]]
    lines += ["", "## 대기 — 아직 로컬 예약", ""] + [f"- {tag}" for tag in state["pending"]]
    if not state["pending"]:
        lines += ["없음."]
    lines += ["", "## 추가 원격 이관으로 명시적으로 해제된 작업", ""]
    lines += [f"- {tag}" for tag in state["remote_fallback_jobs"]] or ["없음."]
    lines += ["", "## 완료 시각 추정", "", "| 작업 | 구분 | 예상 완료 KST |", "|---|---|---|"]
    lines += [f"| {f['job']} | {f['kind']} | {f['finish_kst']} |" for f in forecasts]
    if state["state"] == "complete":
        lines += ["", "로컬 18회는 모두 완료돼 남은 시간 추정이 필요하지 않습니다. 다른 환경 배정 완료를 뜻하지는 않습니다."]
    lines += ["", "예상은 최근 step 속도와 실제 완료 run의 후처리 시간에 따른 대기열 계산이며 보장 시각이 아닙니다. 18:05 이후 새 run을 시작하지 않고 22시에 이 실험의 로컬 작업을 종료하는 기존 설정을 유지합니다.",
              "", "이 문서는 게시 시점 스냅샷입니다. 실제 이관 시에는 원래 서버의 최신 live_handoff.json에서 remote_fallback_jobs를 확인하고 pending 작업과 중복 실행하지 않습니다.", ""]
    (args.output / "PROGRESS_KO.md").write_text("\n".join(lines))
    print(json.dumps({"as_of_kst": timestamp, "completed": [m["tag"] for m in completed_metrics],
                      "comparisons": comparisons, "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
