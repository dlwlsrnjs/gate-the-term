"""Sequential external evaluation of assigned, complete primary checkpoints."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--train-root", type=Path, default=ROOT / "outputs/camera_ready")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/expanded")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--arms")
    parser.add_argument("--include-base", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    protocol = json.loads((HERE / "protocol.json").read_text())
    arms = args.arms.split(",") if args.arms else protocol["primary"]
    seeds = [int(value) for value in args.seeds.split(",")]
    assert set(arms) <= set(protocol["primary"]) and len(arms) == len(set(arms))
    assert set(seeds) <= set(protocol["seeds"]) and len(seeds) == len(set(seeds))
    jobs = ([("base", None)] if args.include_base else []) + [
        (f"{arm}_s{seed}_n100", args.train_root / f"{arm}_s{seed}_n100/adapter") for seed in seeds for arm in arms]
    lock = None
    if not args.dry_run:
        lock = (Path(tempfile.gettempdir()) / f"gate-the-term-{os.getuid()}-gpu{args.gpu}.lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source_sha = hashlib.sha256((HERE / "evaluate_expanded.py").read_bytes()).hexdigest()
    dataset_sha = hashlib.sha256((HERE / "evaluation_data/manifest.json").read_bytes()).hexdigest()
    for tag, adapter in jobs:
        dest = args.output / tag
        if dest.exists():
            result = json.loads((dest / "result.json").read_text())
            cfg = json.loads((dest / "config.json").read_text())
            assert result["complete"] and not result["functionality_smoke"]
            assert cfg["source_sha256"] == source_sha and cfg["dataset_manifest_sha256"] == dataset_sha
            assert cfg["model_revision"] == protocol["model_revision"] and cfg["batch_size"] == 16
            if adapter is not None:
                assert cfg["adapter_sha256"] == {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in adapter.glob("*.safetensors")}
            print("SKIP complete", tag, flush=True)
            continue
        if adapter is not None and not args.dry_run:
            result = json.loads((adapter.parent / "result.json").read_text())
            cfg = json.loads((adapter.parent / "config.json").read_text())
            assert result["steps"] == 100 and cfg["eval_n"] == 60
            assert cfg["source_sha256"] == hashlib.sha256((ROOT / "rebuttal_3213/train_controls.py").read_bytes()).hexdigest()
        command = [sys.executable, "-u", str(HERE / "evaluate_expanded.py"),
                   "--gpu", str(args.gpu), "--output", str(dest.resolve())]
        if adapter is not None:
            command += ["--adapter", str(adapter.resolve())]
        print("EVALUATE", tag, flush=True)
        if args.dry_run:
            print(" ".join(command))
            continue
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / f"{tag}.log").open("x") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
