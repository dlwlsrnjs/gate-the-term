"""Pack, download, and hash-check complete camera-ready training artifacts.

No credentials are stored by this script. Base-model weights are fetched separately
at the pinned publisher revision. Partial/optimizer-state resume is not supported.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def verify(directory, expected_manifest=None):
    manifest_path = directory / "ARTIFACT_MANIFEST.json"
    if expected_manifest:
        assert sha256(manifest_path) == expected_manifest, "Manifest SHA256 mismatch"
    manifest = json.loads(manifest_path.read_text())
    for name, entry in manifest["files"].items():
        parts = PurePosixPath(name)
        assert not parts.is_absolute() and ".." not in parts.parts
        path = directory / name
        assert path.is_file() and not path.is_symlink(), f"Missing regular file: {name}"
        assert path.stat().st_size == entry["bytes"], f"Size mismatch: {name}"
        assert sha256(path) == entry["sha256"], f"SHA256 mismatch: {name}"
    assert len(manifest["runs"]) == 18
    for tag, entry in manifest["runs"].items():
        run = directory / "runs" / tag
        result = json.loads((run / "result.json").read_text())
        cfg = json.loads((run / "config.json").read_text())
        adapter = json.loads((run / "adapter/adapter_config.json").read_text())
        assert result["steps"] == cfg["steps"] == 100 and cfg["eval_n"] == 60
        assert set(result["eval_end"]) == {"bn", "te", "th"}
        assert all(len(d["per_problem"]) == 60 for d in result["eval_end"].values())
        assert adapter["base_model_name_or_path"] == manifest["base_model"]["repo_id"]
        assert adapter["revision"] == manifest["base_model"]["revision"]
        assert manifest["files"][f"runs/{tag}/adapter/adapter_model.safetensors"]["sha256"] == entry["original_weight_sha256"]
    print(json.dumps({"status": "verified", "runs": 18, "files": len(manifest["files"]),
                      "manifest_sha256": sha256(manifest_path)}), flush=True)
    return manifest


def pack(args):
    protocol = json.loads((HERE / "protocol.json").read_text())
    jobs = [f"{arm}_s{seed}_n100" for seed in protocol["seeds"] for arm in protocol["primary"]]
    assert len(jobs) == 18
    args.output.mkdir(parents=True, exist_ok=False)
    original_weights = {}
    # The source run directories remain unchanged. Only copied adapter metadata
    # gets a portable base-model identifier; originals stay beside the copies.
    for tag in jobs:
        source, target = args.train_root / tag, args.output / "runs" / tag
        result = json.loads((source / "result.json").read_text())
        cfg = json.loads((source / "config.json").read_text())
        assert result["steps"] == cfg["steps"] == 100 and cfg["eval_n"] == 60
        assert cfg["source_sha256"] == sha256(ROOT / "rebuttal_3213/train_controls.py")
        assert cfg["coefficients_sha256"] == sha256(ROOT / "rebuttal_3213/control_coefficients.py")
        assert cfg["dropout"] == 0 and cfg["extra"] == 2 and cfg["max_new"] == 320
        assert all(len(d["per_problem"]) == 60 for d in result["eval_end"].values())
        assert not any(f.is_symlink() for f in source.rglob("*"))
        shutil.copytree(source, target)
        original = target / "adapter/adapter_config.json"
        shutil.copy2(original, original.with_name("adapter_config.source.json"))
        portable = json.loads(original.read_text())
        portable.update(base_model_name_or_path=protocol["model_id"], revision=protocol["model_revision"])
        write_json(original, portable)
        card = target / "adapter/README.md"
        if card.exists():
            shutil.copy2(card, card.with_name("README.source.md"))
        card.write_text(f"# {tag}\n\nCompleted 100-step LoRA adapter. Base: {protocol['model_id']} @ {protocol['model_revision']}.\n\nSee the artifact-root README for provenance, loading, and limitations. Weights are unchanged; source metadata is retained.\n")
        weight = target / "adapter/adapter_model.safetensors"
        digest = sha256(weight)
        assert digest == sha256(source / "adapter/adapter_model.safetensors")
        # Validate the safetensors index/offsets without loading tensors or a GPU.
        with weight.open("rb") as handle:
            n = int.from_bytes(handle.read(8), "little")
            header = json.loads(handle.read(n))
        tensors = [value for name, value in header.items() if name != "__metadata__"]
        assert tensors and max(value["data_offsets"][1] for value in tensors) + n + 8 == weight.stat().st_size
        original_weights[tag] = {"arm": cfg["arm"], "seed": cfg["seed"],
                                 "original_weight_sha256": digest, "tensor_count": len(tensors)}
        log = args.train_root.parent / (tag + ".log")
        if log.exists():
            dest = args.output / "logs" / log.name
            dest.parent.mkdir(exist_ok=True)
            shutil.copy2(log, dest)
        print(json.dumps({"packed_run": tag, "weight_bytes": weight.stat().st_size}), flush=True)
    shutil.copytree(HERE / "data", args.output / "data")
    shutil.copytree(HERE / "results/latest", args.output / "results")
    meta = args.output / "metadata"
    meta.mkdir()
    for name in ["protocol.json", "environment_source.json", "source_checksums.json", "allocation_2026-11-26.json", "DATA_SOURCES.md"]:
        shutil.copy2(HERE / name, meta / name)
    shutil.copy2(ROOT / "requirements-camera-ready.txt", meta / "requirements-camera-ready.txt")
    if args.state:
        shutil.copy2(args.state, meta / "final_source_status.json")
    tools = args.output / "tools"
    tools.mkdir()
    shutil.copy2(Path(__file__), tools / "artifacts.py")
    code = args.output / "code"
    code.mkdir()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    subprocess.run(["git", "archive", "--format=tar.gz", "--prefix=gate-the-term/", "--output=" + str((code / "gate-the-term-source.tar.gz").resolve()), commit], cwd=ROOT, check=True)
    (args.output / "README.md").write_text(f"""---
library_name: peft
base_model: {protocol['model_id']}
pipeline_tag: text-generation
tags:
- lora
- grpo
- multilingual
- arithmetic
---

# Gate the Term: complete camera-ready primary cohort

18 final LoRA adapters: six primary arms × seeds 1,2,3, 100 steps each.
Weights occupy about 2.9 GB in total. This is the complete local cohort; it does
not contain the secondary nine runs or the external evaluation of 19 models.

- Code: https://github.com/dlwlsrnjs/gate-the-term
- Code snapshot in this artifact: `{commit}` (`code/gate-the-term-source.tar.gz`).
- Base model: `{protocol['model_id']}`, pinned revision `{protocol['model_revision']}`.
- `runs/<arm>_s<seed>_n100/adapter/`: unchanged safetensors and portable metadata.
- Each run also includes original config, 200 prompt-level training records,
  per-language final predictions, result, and trainer/coefficient source snapshots.
- `logs/`: original stdout logs. `results/`: audited per-seed and three-seed reports.
- `data/`: frozen compressed MGSM and external evaluation inputs, checksums,
  source provenance and available upstream license notices.
- `metadata/`: exact protocol, versions, allocation and source checksums.
- `ARTIFACT_MANIFEST.json` and `SHA256SUMS`: file-level integrity checks.

Source configs retain source-host paths as provenance. Copied adapter configs
replace only base_model_name_or_path and revision for portability; original
adapter metadata is preserved as `adapter_config.source.json` / `README.source.md`.
The original local runs and all weight bytes remain unchanged.

## Download and verify

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="{args.repo_id}", local_dir="received/gate-the-term")
```

Pin the published commit from the GitHub handoff receipt for immutable reproduction.
Run `python received/gate-the-term/tools/artifacts.py verify --directory received/gate-the-term`.
The base model is not duplicated here; prepare.py downloads its fixed official revision.

## Load an adapter

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
model_id = "{protocol['model_id']}"
revision = "{protocol['model_revision']}"
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
base = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, "{args.repo_id}", subfolder="runs/cgh_strict_s1_n100/adapter")
```

For full follow-up commands, see the GitHub camera_ready/ARTIFACT_HANDOFF_KO.md.
External evaluation uses these 18 adapters plus the base model. The secondary
whole_gate/q_resample/component_zvp controls start independently from the fixed
base model, not from a primary-arm adapter. No optimizer/RNG states were saved;
exact mid-training resume is not supported.

## Scientific and data scope

MGSM holds out 60 semantic IDs translated into Bengali, Telugu and Thai; seeds
and translations are not independent new questions. Frozen external selections
contain 2,069 semantic IDs / 4,138 language-item inputs per model and are evaluation-only.
Do not tune on external test results. Three seeds, one 7B model and arithmetic
tasks provide limited generality. Accuracy does not validate an unbiased gradient.
English reference access is privileged training information.

Upstream model and dataset terms remain with their publishers. See
metadata/DATA_SOURCES.md and data/LICENSE.*; no blanket relicensing is asserted.
The official base weights, unrelated/smoke runs, credentials and environments
are not included.
""")
    files = {}
    secret_pattern = re.compile(rb"(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{25,})")
    for path in sorted(args.output.rglob("*")):
        if not path.is_file():
            continue
        name = path.relative_to(args.output).as_posix()
        if path.suffix in {".json", ".jsonl", ".py", ".md", ".txt"}:
            assert not secret_pattern.search(path.read_bytes()), f"Credential-like string in {name}"
        files[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "repo_id": args.repo_id,
                "code_repository": "https://github.com/dlwlsrnjs/gate-the-term", "code_base_commit": commit,
                "base_model": {"repo_id": protocol["model_id"], "revision": protocol["model_revision"]},
                "scope": "18 complete primary runs; secondary/external outcomes not included", "optimizer_rng_saved": False,
                "adapter_metadata_only_portability_edits": True, "runs": original_weights, "files": files}
    write_json(args.output / "ARTIFACT_MANIFEST.json", manifest)
    lines = [f"{entry['sha256']}  {name}" for name, entry in files.items()]
    lines.append(f"{sha256(args.output / 'ARTIFACT_MANIFEST.json')}  ARTIFACT_MANIFEST.json")
    (args.output / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    verify(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    command = sub.add_parser("pack")
    command.add_argument("--train-root", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--repo-id", required=True)
    command.add_argument("--state", type=Path)
    command = sub.add_parser("verify")
    command.add_argument("--directory", type=Path, required=True)
    command.add_argument("--manifest-sha256")
    command = sub.add_parser("download")
    command.add_argument("--directory", type=Path, required=True)
    command.add_argument("--release", type=Path, default=HERE / "artifacts_release.json")
    args = parser.parse_args()
    if args.operation == "pack":
        pack(args)
    elif args.operation == "verify":
        verify(args.directory, args.manifest_sha256)
    else:
        from huggingface_hub import snapshot_download
        release = json.loads(args.release.read_text())
        snapshot_download(repo_id=release["repo_id"], repo_type="model", revision=release["revision"],
                          local_dir=str(args.directory), max_workers=4)
        verify(args.directory, release["manifest_sha256"])


if __name__ == "__main__":
    main()
