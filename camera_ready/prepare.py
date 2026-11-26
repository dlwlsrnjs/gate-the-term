"""Unpack frozen data; optionally download the pinned model or upstream raw data."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROTOCOL = json.loads((HERE / "protocol.json").read_text())


def put(path, payload, expected=None):
    if expected and hashlib.sha256(payload).hexdigest() != expected:
        raise RuntimeError(f"Checksum mismatch: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != payload:
        raise RuntimeError(f"Existing file differs; preserve it before replacing: {path}")
    if not path.exists():
        path.write_bytes(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--model", action="store_true")
    parser.add_argument("--model-directory", type=Path)
    parser.add_argument("--download-raw", action="store_true")
    args = parser.parse_args()
    if args.data_only and (args.model or args.model_directory):
        parser.error("--data-only cannot be combined with a model option")
    for asset in json.loads((HERE / "data/assets_manifest.json").read_text()):
        path = HERE / asset["asset"]
        compressed = path.read_bytes()
        assert hashlib.sha256(compressed).hexdigest() == asset["compressed_sha256"], path
        payload = gzip.decompress(compressed)
        assert hashlib.sha256(payload).hexdigest() == asset["sha256"], path
        if asset["encoding"] == "json-files-gzip":
            values = json.loads(payload)
            assert set(values) == set(asset["files"])
            for name, value in values.items():
                assert Path(name).name == name
                put(ROOT / "safecomp_audit/data/mgsm" / name, value.encode(), asset["files"][name])
        else:
            put(ROOT / asset["destination"], payload, asset["sha256"])
    put(HERE / "evaluation_data/manifest.json", (HERE / "data/evaluation_manifest.json").read_bytes())
    print("Verified frozen MGSM and external data: 60 held-out MGSM IDs; 2,069 new IDs / 4,138 inputs.")
    if args.download_raw:
        for record in json.loads((HERE / "data/source_downloads.json").read_text()):
            target = HERE / "evaluation_data/raw" / record["path"]
            if target.exists():
                payload = target.read_bytes()
            else:
                with urllib.request.urlopen(record["url"], timeout=120) as response:
                    payload = response.read()
            put(target, payload, record["sha256"])
            print("Verified raw source", target.name)
    if args.model or args.model_directory:
        if args.model_directory:
            model = args.model_directory.resolve()
        else:
            os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
            from huggingface_hub import snapshot_download
            model = Path(snapshot_download(PROTOCOL["model_id"], revision=PROTOCOL["model_revision"]))
        assert model.name == PROTOCOL["model_revision"], "Keep the snapshot revision as the directory name"
        for name, expected in json.loads((HERE / "data/model_metadata.json").read_text()).items():
            assert hashlib.sha256((model / name).read_bytes()).hexdigest() == expected, name
        (ROOT / "scripts/model_path.txt").write_text(str(model) + "\n")
        print("Pinned model ready:", model)


if __name__ == "__main__":
    main()
