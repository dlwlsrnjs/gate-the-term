"""Evaluate a saved adapter on frozen external questions, using original scoring.

No pivot, English reference, or gold answer is included in a model input.
--limit-items is for a functionality smoke and is excluded from research tables.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "safecomp_audit"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit-items", type=int)
    args = parser.parse_args()
    marker = HERE.parent / "scripts/model_path.txt"
    model_path = args.model or (Path(marker.read_text().strip()) if marker.exists() else None)
    if model_path is None or not (model_path / "config.json").exists():
        parser.error("Prepare the pinned model first, or pass --model /path/to/snapshot")
    model = str(model_path.resolve())
    assert args.batch_size > 0
    assert args.limit_items is None or args.limit_items > 0
    free = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free",
                                    "--format=csv,noheader,nounits"], text=True).splitlines()
    assert int(free[args.gpu]) >= 24000, "Insufficient free GPU memory; no model loaded"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    import transformers
    import peft
    from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
    from peft import PeftModel
    sys.path.insert(0, str(DATA / "src"))
    from mgsm_grpo import chat, components

    torch.set_num_threads(4)
    set_seed(3213)
    manifest = json.loads((HERE / "evaluation_data/manifest.json").read_text())
    rows = []
    for name in ["indic_gsm8k", "msvamp"]:
        entry = manifest[name]
        path = HERE / entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        values = [json.loads(line) for line in path.read_text().splitlines()]
        if args.limit_items is not None:
            keep = sorted({row["id"] for row in values})[:args.limit_items]
            values = [row for row in values if row["id"] in keep]
        rows.extend(values)
    rows.sort(key=lambda row: (row["dataset"], row["lang"], row["source_index"]))
    args.output.mkdir(parents=True, exist_ok=False)
    adapter_hashes = {}
    if args.adapter:
        for path in sorted(args.adapter.glob("*.safetensors")):
            adapter_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        assert adapter_hashes, "No saved adapter tensors found"
    config = {**vars(args), "model": model, "model_revision": MODEL_REVISION, "max_new_tokens": 320,
              "do_sample": False, "repetition_penalty": 1., "torch": torch.__version__,
              "transformers": transformers.__version__, "peft": peft.__version__,
              "adapter_sha256": adapter_hashes,
              "dataset_manifest_sha256": hashlib.sha256((HERE / "evaluation_data/manifest.json").read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scoring_sha256": hashlib.sha256((DATA / "src/mgsm_grpo.py").read_bytes()).hexdigest(),
              "is_functionality_smoke": args.limit_items is not None}
    (args.output / "config.json").write_text(json.dumps(config, default=str, indent=2) + "\n")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model, local_files_only=True,
                                               torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    if args.adapter:
        model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
    model = model.to("cuda").eval()
    results = []
    started = time.time()
    groups = defaultdict(list)
    for row in rows:
        groups[row["dataset"], row["lang"]].append(row)
    with (args.output / "predictions.jsonl").open("x") as log:
        for (dataset, lang), group in groups.items():
            for start in range(0, len(group), args.batch_size):
                batch = group[start:start + args.batch_size]
                encoded = tokenizer([chat(tokenizer, row["q"]) for row in batch],
                                    padding=True, return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    sequences = model.generate(**encoded, max_new_tokens=320, do_sample=False,
                                               repetition_penalty=1., pad_token_id=tokenizer.pad_token_id,
                                               use_cache=True)
                outputs = sequences[:, encoded.input_ids.shape[1]:]
                texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                for row, text in zip(batch, texts):
                    result = {"dataset": dataset, "language": lang, "id": row["id"],
                              "source_index": row["source_index"], "text": text,
                              **components(text, row["gold"], lang)}
                    results.append(result)
                    log.write(json.dumps(result, ensure_ascii=False) + "\n")
                log.flush()
                print(json.dumps({"dataset": dataset, "lang": lang,
                                  "completed": min(start + len(batch), len(group)), "n": len(group),
                                  "elapsed_s": time.time() - started}), flush=True)
    summary = {}
    for dataset, lang in groups:
        selected = [row for row in results if row["dataset"] == dataset and row["language"] == lang]
        summary[f"{dataset}/{lang}"] = {"n": len(selected), **{
            key: sum(row[key] for row in selected) / len(selected) for key in ["corr", "corr_raw", "lang"]}}
    (args.output / "result.json").write_text(json.dumps({"complete": True,
        "functionality_smoke": args.limit_items is not None, "by_dataset_language": summary,
        "elapsed_s": time.time() - started}, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
