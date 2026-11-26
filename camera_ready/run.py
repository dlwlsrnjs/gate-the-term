"""Run assigned camera-ready jobs using the unchanged common training engine."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--plan", choices=["primary", "secondary", "all"], default="primary")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--arms")
    parser.add_argument("--assignment", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/camera_ready")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    protocol = json.loads((HERE / "protocol.json").read_text())
    arms = protocol["primary"] if args.plan == "primary" else protocol["secondary"] if args.plan == "secondary" else protocol["primary"] + protocol["secondary"]
    if args.arms:
        selected = args.arms.split(",")
        assert set(selected) <= set(arms) and len(selected) == len(set(selected))
        arms = selected
    seeds = [int(value) for value in args.seeds.split(",")]
    assert set(seeds) <= set(protocol["seeds"]) and len(seeds) == len(set(seeds))
    jobs = [f"{arm}_s{seed}_n100" for seed in seeds for arm in arms]
    if args.assignment:
        assignment = json.loads(args.assignment.read_text())
        assert assignment["source_managers_paused"] is True, "Pause source managers before dispatching their pending jobs"
        assert assignment["training_engine_sha256"] == hashlib.sha256((ROOT / "rebuttal_3213/train_controls.py").read_bytes()).hexdigest()
        jobs = [job for job in jobs if job in assignment["remote_jobs"]]
    if not jobs:
        raise SystemExit("No jobs match this plan/assignment")
    lock = None
    if not args.dry_run:
        lock = (Path(tempfile.gettempdir()) / f"gate-the-term-{os.getuid()}-gpu{args.gpu}.lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for job in jobs:
        match = re.fullmatch(r"(.+)_s([123])_n100", job)
        assert match
        arm, seed = match.groups()
        command = [sys.executable, "-u", str(ROOT / "scripts/run_suite.py"), "--gpu", str(args.gpu),
                   "--arms", arm, "--seeds", seed, "--eval-n", "60", "--output", str(args.output.resolve())]
        if args.dry_run:
            command.append("--dry-run")
        print("JOB", job, flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
