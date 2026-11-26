"""Copy complete training result directories, rejecting conflicting duplicate jobs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--with-adapters", action="store_true")
    args = parser.parse_args()
    protocol = json.loads((HERE / "protocol.json").read_text())
    expected = hashlib.sha256((ROOT / "rebuttal_3213/train_controls.py").read_bytes()).hexdigest()
    for root in args.roots:
        for source in sorted(root.iterdir()):
            if not source.is_dir() or not re.fullmatch(r".+_s[123]_n100", source.name):
                continue
            if not (source / "result.json").exists():
                print("INCOMPLETE, preserved on source:", source)
                continue
            cfg = json.loads((source / "config.json").read_text())
            result = json.loads((source / "result.json").read_text())
            assert source.name in protocol["jobs"] and cfg["eval_n"] == 60 and cfg["steps"] == 100
            assert cfg["source_sha256"] == expected and cfg["dropout"] == 0
            assert all(len(value["per_problem"]) == 60 for value in result["eval_end"].values())
            files = [path for path in source.rglob("*") if path.is_file() and (args.with_adapters or "adapter" not in path.relative_to(source).parts)]
            dest = args.destination / source.name
            for path in files:
                other = dest / path.relative_to(source)
                if other.exists() and hashlib.sha256(other.read_bytes()).digest() != hashlib.sha256(path.read_bytes()).digest():
                    raise RuntimeError(f"Conflicting duplicate job; neither version overwritten: {other}")
            for path in files:
                other = dest / path.relative_to(source)
                other.parent.mkdir(parents=True, exist_ok=True)
                if not other.exists():
                    shutil.copy2(path, other)
            print("MERGED", source.name)


if __name__ == "__main__":
    main()
