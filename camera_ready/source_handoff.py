"""Inspect/pause this chat's original finite queues and reserve active child jobs.

Pausing managers does not interrupt current training children. No model/RNG resume
is claimed. Run on the source host, then transfer the emitted assignment JSON.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time


def verify_process(pid, script):
    path = Path(f"/proc/{pid}/cmdline")
    if not path.exists() or str(script.resolve()).encode() not in path.read_bytes():
        raise RuntimeError(f"PID {pid} is not the expected queue: {script}")


def pause(pid, script):
    verify_process(pid, script)
    os.kill(pid, signal.SIGSTOP)
    for _ in range(20):
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].strip().split()[0]
        if state in ["T", "t"]:
            return
        time.sleep(.05)
    raise RuntimeError(f"Queue {pid} did not enter the stopped state")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--pause-source", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("source_assignment.json"))
    args = parser.parse_args()
    root = args.source_root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    state = json.loads((root / "status.json").read_text())
    if state["state"] != "running":
        raise RuntimeError("This handoff supports an active training queue; inspect finished results separately")
    expanded = json.loads((root / "expanded_evaluation/status.json").read_text())
    if expanded["state"] not in ["waiting_for_training", "assigned_remote"]:
        raise RuntimeError("External evaluation is already active/finished; do not use this training-only handoff")
    print("Source active jobs:", [v["job"] for v in state["workers"].values() if v["state"] == "running"])
    if not args.pause_source:
        print("Inspection only. Add --pause-source to pause managers and write a transferable assignment.")
        return
    if args.output.exists():
        raise RuntimeError("Assignment output exists; preserve it rather than overwriting")
    # Verify both targets before changing either process. Only manager PIDs are signalled.
    manager = Path(state.get("manager_script", root / "run_suite.py"))
    assert manager.name in ["run_suite.py", "run_parallel.py"]
    verify_process(state["pid"], manager)
    if expanded["state"] == "waiting_for_training":
        verify_process(expanded["pid"], root / "run_expanded_evaluation.py")
        pause(expanded["pid"], root / "run_expanded_evaluation.py")
    pause(state["pid"], manager)
    state = json.loads((root / "status.json").read_text())
    jobs = state.get("assigned_jobs") or [f"{arm}_s{seed}_n100" for seed in manifest["seeds"]
                                          for arm in manifest["primary"] + manifest["secondary"]]
    completed = [job for job in jobs if (root / "runs" / job / "result.json").exists()]
    active = []
    for worker in state["workers"].values():
        if worker["state"] != "running" or worker["job"] in completed:
            continue
        cmd = Path(f"/proc/{worker['pid']}/cmdline")
        if cmd.exists() and b"train_controls.py" in cmd.read_bytes():
            active.append(worker["job"])
    engine = next(value for name, value in manifest["source_and_data_sha256"].items() if name.endswith("/train_controls.py"))
    assignment = {"source_managers_paused": True, "training_engine_sha256": engine,
                  "source_root": str(root), "training_manager_pid": state["pid"],
                  "evaluation_manager_pid": expanded["pid"], "completed_on_source": completed,
                  "reserved_active_on_source": active,
                  "remote_jobs": [job for job in jobs if job not in completed + active],
                  "notes": ["Current source training children continue to completion.",
                            "Transfer their complete result directories and adapters after they finish.",
                            "Keep the old managers paused while remote workers own these jobs.",
                            "After merging remote results, use the portable launchers instead of resuming the old manager.",
                            "Incomplete jobs start from step zero; adapters do not contain optimizer/RNG state."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(assignment, indent=2) + "\n")
    print("Wrote", args.output, "remote jobs:", len(assignment["remote_jobs"]), "reserved source jobs:", active)


if __name__ == "__main__":
    main()
