"""Summarize the complete external cohort, clustering by source item and seed."""
import argparse, json
from pathlib import Path
from prepare import PROTOCOL
PRIMARY = PROTOCOL["primary"]
SEEDS = PROTOCOL["seeds"]

def atomic_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")

def summarize():
    import numpy as np
    data, configs = {}, []
    for tag in ["base"] + [f"{arm}_s{seed}_n100" for seed in SEEDS for arm in PRIMARY]:
        root = OUT / tag
        result = json.loads((root / "result.json").read_text())
        assert result["complete"] and not result["functionality_smoke"]
        configs.append(json.loads((root / "config.json").read_text()))
        data[tag] = {(row["dataset"], row["language"], row["id"]): row
                     for row in map(json.loads, (root / "predictions.jsonl").read_text().splitlines())}
    reference = configs[0]
    fixed = ["model_revision", "batch_size", "max_new_tokens", "do_sample", "repetition_penalty",
             "torch", "transformers", "peft", "dataset_manifest_sha256", "source_sha256", "scoring_sha256"]
    assert all(all(cfg[k] == reference[k] for k in fixed) for cfg in configs)
    assert all(sorted(rows) == sorted(data["base"]) for rows in data.values())
    report = {"seeds": SEEDS, "scope": "Exploratory external evaluation; fixed datasets and languages.",
              "by_dataset": {}, "comparisons": []}
    table = ["% Generated only after every primary arm/seed completes external evaluation.",
             r"\begin{tabular}{llrrr}", r"\toprule",
             r"Dataset & Arm & Accuracy & Raw accuracy & Language \\", r"\midrule"]
    for dataset in ["indic_gsm8k", "msvamp"]:
        keys = sorted(key for key in data["base"] if key[0] == dataset)
        ids = sorted({key[2] for key in keys})
        langs = sorted({key[1] for key in keys})
        dataset_report = {"distinct_semantic_items": len(ids), "languages": langs, "arms": {}}
        for arm in ["base"] + PRIMARY:
            tags = ["base"] if arm == "base" else [f"{arm}_s{s}_n100" for s in SEEDS]
            values = {field: float(np.mean([data[tag][key][field] for tag in tags for key in keys]))
                      for field in ["corr", "corr_raw", "lang"]}
            values["by_language"] = {lang: {field: float(np.mean([
                data[tag][key][field] for tag in tags for key in keys if key[1] == lang]))
                for field in ["corr", "corr_raw", "lang"]} for lang in langs}
            dataset_report["arms"][arm] = values
            table.append(f"{dataset.replace('_', chr(92) + '_')} & {arm.replace('_', chr(92) + '_')} & "
                         f"{values['corr']:.4f} & {values['corr_raw']:.4f} & {values['lang']:.4f}" + r" \\")
        report["by_dataset"][dataset] = dataset_report
        for left, right in [("vanilla", "cgh_legacy"), ("vanilla", "split"), ("split", "cgh_strict"),
                            ("pivot_rsft", "cgh_strict"), ("cgh_strict_nois", "cgh_strict"),
                            ("cgh_legacy", "cgh_strict")]:
            delta = np.array([[np.mean([
                data[f"{right}_s{seed}_n100"][dataset, lang, item]["corr"] -
                data[f"{left}_s{seed}_n100"][dataset, lang, item]["corr"]
                for lang in langs]) for item in ids] for seed in SEEDS])
            rng = np.random.default_rng(3213)
            boots = [float(delta[np.ix_(rng.integers(len(SEEDS), size=len(SEEDS)),
                                        rng.integers(len(ids), size=len(ids)))].mean())
                     for _ in range(20000)]
            report["comparisons"].append({"dataset": dataset, "left": left, "right": right,
                                          "delta": float(delta.mean()), "per_seed": delta.mean(1).tolist(),
                                          "exploratory_crossed_95ci": np.quantile(boots, [.025, .975]).tolist()})
    table.extend([r"\bottomrule", r"\end{tabular}"])
    atomic_json(OUT / "summary.json", report)
    (OUT / "tab_expanded.tex").write_text("\n".join(table) + "\n")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    a=p.parse_args()
    OUT=a.root.resolve()
    summarize()
    print("Wrote",OUT/"summary.json", "and",OUT/"tab_expanded.tex")
