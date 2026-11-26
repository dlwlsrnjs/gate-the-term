"""Summarize the complete primary cohort; crossed seed/problem bootstrap.

Consumes validated report_completed.py exports. Refuses incomplete primary arms.
Translations are fixed strata; seed repetitions are not new independent items.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
LANGS = ["bn", "te", "th"]
PAIRS = [("vanilla", "split"), ("vanilla", "cgh_strict"), ("vanilla", "pivot_rsft"),
         ("vanilla", "cgh_legacy"), ("vanilla", "cgh_strict_nois"),
         ("split", "cgh_strict"), ("split", "pivot_rsft"),
         ("pivot_rsft", "cgh_strict"), ("pivot_rsft", "cgh_strict_nois"),
         ("cgh_strict_nois", "cgh_strict"), ("cgh_legacy", "cgh_strict")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=HERE / "results/latest")
    args = parser.parse_args()
    protocol = json.loads((HERE / "protocol.json").read_text())
    completed = json.loads((args.directory / "completed_results.json").read_text())
    seeds, arms = protocol["seeds"], protocol["primary"]
    assert seeds == [1, 2, 3]
    metrics = {(r["arm"], r["seed"]): r for r in completed["completed_runs"]}
    assert all((arm, seed) in metrics for arm in arms for seed in seeds)
    with (args.directory / "completed_items.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    values = {(r["arm"], int(r["seed"]), r["language"], int(r["semantic_id"])):
              {field: float(r[field]) for field in ["corr", "corr_raw", "lang"]} for r in rows}
    assert len(values) == len(rows)
    ids = sorted({int(r["semantic_id"]) for r in rows})
    assert len(ids) == 60
    arrays = {arm: {field: np.array([[[values[arm, seed, lg, item][field] for lg in LANGS]
                                     for item in ids] for seed in seeds])
                    for field in ["corr", "corr_raw", "lang"]} for arm in arms}
    summaries = {}
    for arm in arms:
        a = arrays[arm]
        assert [int(x) for x in a["corr"].sum(axis=(1, 2))] == [metrics[arm, s]["overall"]["corr"] for s in seeds]
        assert np.isclose(a["corr"].mean(), completed["method_three_seed_means"][arm])
        summaries[arm] = {"per_seed_target_counts": a["corr"].sum(axis=(1, 2)).astype(int).tolist(),
                          "mean_target_accuracy": float(a["corr"].mean()),
                          "mean_raw_accuracy": float(a["corr_raw"].mean()),
                          "mean_script_pass": float(a["lang"].mean()),
                          "target_delta_vs_grpo_pp": float((a["corr"] - arrays["vanilla"]["corr"]).mean() * 100),
                          "by_language": {lg: {field: float(a[field][:, :, j].mean()) for field in a}
                                          for j, lg in enumerate(LANGS)}}
    comparisons = []
    for left, right in PAIRS:
        # Mean over selected languages first; same seed/ID resamples for both arms.
        delta = (arrays[right]["corr"] - arrays[left]["corr"]).mean(axis=2)
        rng = np.random.default_rng(3213)
        boots = [float(delta[np.ix_(rng.integers(len(seeds), size=len(seeds)),
                                    rng.integers(len(ids), size=len(ids)))].mean()) for _ in range(20000)]
        comparisons.append({"left": left, "right": right, "delta_pp": float(delta.mean() * 100),
                            "per_seed_delta_pp": (delta.mean(axis=1) * 100).tolist(),
                            "exploratory_crossed_95ci_pp": (np.quantile(boots, [.025, .975]) * 100).tolist()})
    report = {"as_of_kst": completed["as_of_kst"], "scope": "Complete local primary cohort; not the external or secondary cohort.",
              "seeds": seeds, "semantic_ids": ids, "languages": LANGS, "completed_runs": len(arms) * len(seeds),
              "bootstrap": {"replicates": 20000, "rng_seed": 3213, "method": "Independently resample 3 seeds and 60 semantic IDs; same crossed indices for both arms; keep translations together.",
                            "limits": "Exploratory percentile intervals with only 3 seeds; languages/model/dataset fixed; no multiplicity adjustment."},
              "input_completed_results_sha256": hashlib.sha256((args.directory / "completed_results.json").read_bytes()).hexdigest(),
              "input_items_sha256": hashlib.sha256((args.directory / "completed_items.csv").read_bytes()).hexdigest(),
              "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "arms": summaries, "comparisons": comparisons}
    (args.directory / "three_seed_results.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# 로컬 핵심 18회 완료 — 3-seed 결과", "", f"기준 시각: {completed['as_of_kst']}", "",
             "핵심 6조건 × seeds 1,2,3의 100-step 학습과 최종 MGSM 평가가 모두 완료됐습니다. 다른 환경의 후순위 학습·확장 평가 결과를 포함하지 않습니다.", "",
             "| 조건 | Seed 1 /180 | Seed 2 /180 | Seed 3 /180 | 3-seed 평균 target | GRPO 대비 | Raw 평균 | Script 통과 평균 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, m in summaries.items():
        lines.append(f"| {arm} | " + " | ".join(str(n) for n in m["per_seed_target_counts"])
                     + f" | {m['mean_target_accuracy']:.2%} | {m['target_delta_vs_grpo_pp']:+.2f} pp | {m['mean_raw_accuracy']:.2%} | {m['mean_script_pass']:.2%} |")
    lines += ["", "## 언어별 3-seed 평균", "",
              "각 언어는 같은 원문 60개를 세 seed에서 평가했습니다. 180개 독립 문제를 뜻하지 않습니다.", "",
              "| 조건 | Bengali | Telugu | Thai |", "|---|---:|---:|---:|"]
    for arm, m in summaries.items():
        lines.append(f"| {arm} | " + " | ".join(f"{m['by_language'][lg]['corr']:.2%}" for lg in LANGS) + " |")
    lines += ["", "## 짝지은 차이와 seed·원문 ID 불확실성", "",
              "| 비교 | Seed 1 Δ pp | Seed 2 Δ pp | Seed 3 Δ pp | 평균 Δ pp | 탐색적 crossed 95% 구간 pp |",
              "|---|---:|---:|---:|---:|---:|"]
    for c in comparisons:
        low, high = c["exploratory_crossed_95ci_pp"]
        lines.append(f"| {c['right']} − {c['left']} | " + " | ".join(f"{v:+.2f}" for v in c["per_seed_delta_pp"])
                     + f" | {c['delta_pp']:+.2f} | [{low:+.2f}, {high:+.2f}] |")
    lines += ["", "20,000회 bootstrap에서 seed와 원문 ID를 독립적으로 resample하고, 같은 원문의 세 번역과 두 비교 조건을 함께 유지했습니다. 기존 단일-seed 문항 구간과 다릅니다. 3개 seed의 경험분포만 사용한 탐색적 구간으로 학습 변동 추정이 제한적이며 다중 비교 보정은 없습니다.", "",
              "## 적용 범위", "", "- Target은 숫자 정답과 script 기준을 함께 통과한 값입니다. Raw 정답·script 통과율은 별도 보고합니다.",
              "- 원문은 60개이며 3언어·3seed 반복을 새로운 독립 문제로 세지 않습니다. 기존 논문의 역사적 cohort와 합치지 않습니다.",
              "- 한 base model, 100 steps, privileged English reference access라는 제한을 유지합니다. 성능 비교는 exact gradient 복원이나 unbiasedness의 검증이 아닙니다.",
              "- 다른 환경 배정인 학습 9회와 외부 2,069개 원문/4,138개 입력 × 19개 모델 평가는 이 로컬 완료와 별도입니다.", "",
              "[문항별 CSV](completed_items.csv) · [원본 집계·출처](completed_results.json) · [3-seed JSON](three_seed_results.json) · [진행](PROGRESS_KO.md)", "",
              "재생성: report_completed.py 실행 후 python camera_ready/report_three_seed.py를 실행합니다.", ""]
    (args.directory / "THREE_SEED_RESULTS_KO.md").write_text("\n".join(lines))
    print(json.dumps({"arms": summaries, "comparisons": comparisons}, ensure_ascii=False))


if __name__ == "__main__":
    main()
