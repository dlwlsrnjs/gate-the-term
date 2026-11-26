"""Check exact runtime versions, sources, data, alignment, and optional GPU/model."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    assert sys.version_info[:2] == (3, 11), "Python 3.11 is required for this cohort"
    environment = json.loads((HERE / "environment_source.json").read_text())
    for name, expected in environment["packages"].items():
        actual = metadata.version(name)
        assert actual == expected, f"{name}: expected {expected}, found {actual}"
    print("Exact Python/package versions PASS")
    for name, expected in json.loads((HERE / "source_checksums.json").read_text()).items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    for name, expected in json.loads((ROOT / "scripts/data_checksums.json").read_text()).items():
        assert hashlib.sha256((ROOT / "safecomp_audit/data/mgsm" / name).read_bytes()).hexdigest() == expected, name
    external = json.loads((HERE / "evaluation_data/manifest.json").read_text())
    for name in ["indic_gsm8k", "msvamp"]:
        entry = external[name]
        path = HERE / entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == entry["language_item_pairs"]
        assert len({row["id"] for row in rows}) == entry["distinct_semantic_items"]
    sys.path.insert(0, str(ROOT / "safecomp_audit/src"))
    from mgsm_grpo import load_parallel
    per, train, heldout = load_parallel(["bn", "te", "th"])
    assert len(train) == 190 and len(heldout) == 60 and not set(train) & set(heldout)
    assert all(len({language[i]["gold"] for language in per.values()}) == 1 for i in range(250))
    print("Sources, frozen data, semantic alignment, and 190/60 split PASS")
    if not args.cpu_only:
        import torch
        assert torch.cuda.is_available() and torch.cuda.get_device_properties(args.gpu).major >= 8
        free = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).splitlines()
        assert int(free[args.gpu]) >= 24000
        marker = ROOT / "scripts/model_path.txt"
        model = Path(marker.read_text().strip())
        assert model.name == json.loads((HERE / "protocol.json").read_text())["model_revision"]
        for name, expected in json.loads((HERE / "data/model_metadata.json").read_text()).items():
            assert hashlib.sha256((model / name).read_bytes()).hexdigest() == expected
        print("GPU headroom and pinned local model PASS")


if __name__ == "__main__":
    main()
