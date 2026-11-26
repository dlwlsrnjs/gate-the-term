"""Freeze external evaluation data; exclude all 250 MGSM source questions.

Public source revisions and file SHA256s are in evaluation_data/raw/.
Only target-language questions are model inputs. English is retained for overlap
checks and provenance. No generated answers are inspected when selecting items.
"""
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import unicodedata
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
RAW = HERE / "evaluation_data/raw"
OUT = HERE / "evaluation_data"
MGSM = HERE.parent / "safecomp_audit/data/mgsm/mgsm_en.tsv"


def canonical(text):
    return "".join(char for char in unicodedata.normalize("NFKC", text).casefold() if char.isalnum())


def sid(text):
    return hashlib.sha256(canonical(text).encode()).hexdigest()


def gold_number(text):
    value = Decimal(str(text).replace(",", "").strip())
    assert value.is_finite()
    return str(value.normalize())


def write_dataset(name, rows):
    seen = {}
    duplicates = 0
    for row in rows:
        key = row["id"], row["lang"]
        if key in seen:
            assert seen[key]["gold"] == row["gold"], "Conflicting labels for the same question"
            duplicates += 1
            continue
        assert row["q"].strip() and row["source_english"].strip()
        seen[key] = row
    selected = list(seen.values())
    path = OUT / f"{name}.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))
    ids = {row["id"] for row in selected}
    languages = sorted({row["lang"] for row in selected})
    assert all({row["id"] for row in selected if row["lang"] == lang} == ids for lang in languages)
    return {"path": path.relative_to(HERE).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "distinct_semantic_items": len(ids), "language_item_pairs": len(selected),
            "by_language": dict(Counter(row["lang"] for row in selected)),
            "duplicate_language_item_rows_removed": duplicates}, ids


def main():
    overlap = {canonical(line.split("\t")[0]) for line in MGSM.read_text().splitlines()}
    assert len(overlap) == 250
    indic = pq.read_table(RAW / "indic_test.parquet", columns=[
        "question", "answer", "id", "question_Bengali_translation", "question_Telugu_translation"]).to_pylist()
    assert len(indic) == 1319
    matched = {canonical(row["question"]) for row in indic} & overlap
    assert matched == overlap, "All MGSM source questions must be explicitly accounted for"
    indic_rows = []
    for index, row in enumerate(indic):
        if canonical(row["question"]) in overlap:
            continue
        gold = gold_number(row["answer"].rsplit("####", 1)[1])
        for lang, field in [("bn", "question_Bengali_translation"), ("te", "question_Telugu_translation")]:
            indic_rows.append({"dataset": "indic_gsm8k", "id": sid(row["question"]),
                               "source_index": index, "source_id": row["id"], "lang": lang,
                               "q": row[field], "gold": gold, "source_english": row["question"]})
    indic_report, indic_ids = write_dataset("indic_gsm8k_bn_te", indic_rows)
    msvamp = {name: [json.loads(line) for line in (RAW / f"msvamp_{name}.json").read_text().splitlines() if line.strip()]
              for name in ["English", "Bengali", "Thai"]}
    assert all(len(rows) == 1000 for rows in msvamp.values())
    msvamp_rows, exclusions = [], 0
    for index, source in enumerate(msvamp["English"]):
        english = source["m_query"]
        for name in ["Bengali", "Thai"]:
            assert canonical(msvamp[name][index]["query"]) == canonical(english)
            assert gold_number(msvamp[name][index]["response"]) == gold_number(source["response"])
        if canonical(english) in overlap or sid(english) in indic_ids:
            exclusions += 1
            continue
        for lang, name in [("bn", "Bengali"), ("th", "Thai")]:
            msvamp_rows.append({"dataset": "msvamp", "id": sid(english), "source_index": index,
                                "lang": lang, "q": msvamp[name][index]["m_query"],
                                "gold": gold_number(source["response"]), "source_english": english})
    msvamp_report, msvamp_ids = write_dataset("msvamp_bn_th", msvamp_rows)
    assert not indic_ids & msvamp_ids
    report = {"indic_gsm8k": indic_report, "msvamp": msvamp_report,
              "mgsm_source_questions_excluded_from_indic": len(matched),
              "overlap_questions_excluded_from_msvamp": exclusions,
              "total_distinct_semantic_items": len(indic_ids | msvamp_ids),
              "total_language_item_pairs": indic_report["language_item_pairs"] + msvamp_report["language_item_pairs"],
              "limitations": ["External translated benchmarks; translation provenance differs from human-translated MGSM.",
                              "MSVAMP release has no Telugu files; do not report a Telugu result for it.",
                              "Both are arithmetic evaluations; this is not evidence of non-math repair training.",
                              "Normalized exact English overlap is excluded; near duplicates and pretraining contamination remain possible.",
                              "Translations of one source item are clustered for inference; seeds do not add independent questions.",
                              "External test sets are evaluation-only and must not be used for training or hyperparameter selection."]}
    (OUT / "manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
