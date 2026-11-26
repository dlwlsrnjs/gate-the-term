"""Assign only jobs explicitly released by the local deadline scheduler."""
import argparse
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-state", type=Path, required=True)
    parser.add_argument("--allocation", type=Path, default=HERE / "allocation_2026-11-26.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state = json.loads(args.source_state.read_text())
    allocation = json.loads(args.allocation.read_text())
    jobs = state["remote_fallback_jobs"]
    assert set(jobs) <= set(allocation["local_training_jobs"])
    assert not set(jobs) & set(state["completed"])
    assert not set(jobs) & {value["job"] for value in state["workers"].values()}
    if not jobs:
        raise SystemExit("No local jobs have been released; do not run still-queued local jobs remotely")
    if args.output.exists():
        raise RuntimeError("Preserve the previous assignment; choose a new output filename")
    allocation["local_training_jobs"] = []
    allocation["remote_training_jobs"] = jobs
    allocation["released_from_source_state"] = state["updated_utc"]
    allocation["notes"].append("Only explicitly released local jobs; original remote jobs are excluded.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(allocation, indent=2) + "\n")
    print("Released remote training jobs:", jobs)

if __name__ == "__main__":
    main()
