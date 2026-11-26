"""Run unchanged experiment jobs concurrently, with a local hard deadline.

Only scheduling changes: G, prompt batches, optimizer, seeds, and scoring stay fixed.
Adopted jobs keep their existing processes. Incomplete deadline jobs are preserved
and listed for a full restart on another host, since optimizer/RNG resume is absent.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def atomic(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, default=str) + "\n")
    temp.replace(path)


def process_alive(pid):
    path = Path(f"/proc/{pid}/stat")
    if not path.exists():
        return False
    return path.read_text().rsplit(")", 1)[1].strip().split()[0] != "Z"


def validate_pid(pid, tag):
    path = Path(f"/proc/{pid}/cmdline")
    if not path.exists():
        return False
    values = path.read_bytes().decode(errors="replace").split("\0")
    match = re.fullmatch(r"(.+)_s([123])_n100", tag)
    if not match or not any(value.endswith("train_controls.py") for value in values):
        return False
    arm, seed = match.groups()
    return all(flag in values and values[values.index(flag) + 1] == value
               for flag, value in [("--arm", arm), ("--seed", seed), ("--steps", "100")])


def gpu_status():
    text = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu",
                                    "--format=csv,noheader,nounits"], text=True)
    return {int(v[0]): {"used_mib": int(v[1]), "free_mib": int(v[2]), "utilization": int(v[3])}
            for v in [[part.strip() for part in line.split(",")] for line in text.splitlines()]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    parser.add_argument("--allocation", type=Path)
    parser.add_argument("--role", choices=["local", "remote"], default="local")
    parser.add_argument("--plan", choices=["primary", "secondary", "all"], default="primary")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/camera_ready")
    parser.add_argument("--state-dir", type=Path, default=ROOT / "outputs/parallel_state")
    parser.add_argument("--adopt-state", type=Path)
    parser.add_argument("--deadline")
    parser.add_argument("--estimated-run-minutes", type=float, default=210)
    parser.add_argument("--finish-margin-minutes", type=float, default=25)
    parser.add_argument("--min-free-mib", type=int, default=24000)
    parser.add_argument("--launch-interval-seconds", type=float, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    assert 1 <= args.workers_per_gpu <= 4, "At most four tested-size replicas per 80GB GPU"
    gpus = [int(value) for value in args.gpus.split(",")]
    assert gpus and len(gpus) == len(set(gpus))
    protocol = json.loads((HERE / "protocol.json").read_text())
    arms = protocol["primary"] if args.plan == "primary" else protocol["secondary"] if args.plan == "secondary" else protocol["primary"] + protocol["secondary"]
    jobs = [f"{arm}_s{seed}_n100" for seed in protocol["seeds"] for arm in arms]
    if args.allocation:
        allocation = json.loads(args.allocation.read_text())
        jobs = allocation[f"{args.role}_training_jobs"]
        if args.role == "local" and not args.deadline:
            args.deadline = allocation["local_deadline"]
    assert len(jobs) == len(set(jobs)) and set(jobs) <= set(protocol["jobs"])
    deadline = datetime.fromisoformat(args.deadline).timestamp() if args.deadline else None
    if deadline is not None:
        assert datetime.fromisoformat(args.deadline).tzinfo is not None
        assert deadline > time.time(), "The local allocation deadline has already passed"
    assert args.estimated_run_minutes > 0 and args.finish_margin_minutes >= 0
    model = (ROOT / "scripts/model_path.txt").read_text().strip()
    engine = ROOT / "rebuttal_3213/train_controls.py"
    engine_sha = hashlib.sha256(engine.read_bytes()).hexdigest()
    coefficients_sha = hashlib.sha256((engine.parent / "control_coefficients.py").read_bytes()).hexdigest()
    active, completed, failed, interrupted, adopted = {}, [], {}, [], []
    if args.adopt_state:
        previous = json.loads(args.adopt_state.read_text())
        for key, worker in previous["workers"].items():
            tag = worker["job"]
            if worker["state"] != "running" or tag not in jobs:
                continue
            if (args.output / tag / "result.json").exists():
                continue
            assert validate_pid(worker["pid"], tag), f"Unexpected adopted PID for {tag}"
            gpu = int(worker.get("gpu", key.split(":")[0]))
            assert gpu in gpus
            active[tag] = {"gpu": gpu, "pid": worker["pid"], "adopted": True, "process": None}
            adopted.append(tag)
    for tag in jobs:
        dest = args.output / tag
        if (dest / "result.json").exists():
            cfg = json.loads((dest / "config.json").read_text())
            assert cfg["source_sha256"] == engine_sha and cfg["coefficients_sha256"] == coefficients_sha
            assert cfg["eval_n"] == 60 and cfg["steps"] == 100
            completed.append(tag)
        elif dest.exists() and tag not in active:
            raise RuntimeError(f"Incomplete unadopted job: {dest}; preserve it before restarting")
    pending = [tag for tag in jobs if tag not in active and tag not in completed]
    print(json.dumps({"jobs": jobs, "adopted": adopted, "pending": pending,
                      "deadline": args.deadline, "workers_per_gpu": args.workers_per_gpu}, indent=2), flush=True)
    if args.dry_run:
        return
    args.state_dir.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    locks = []
    for path in [args.state_dir / ".parallel.lock", *[
        Path(tempfile.gettempdir()) / f"gate-the-term-{os.getuid()}-gpu{gpu}.lock" for gpu in gpus]]:
        lock = path.open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
    atomic(args.state_dir / "parallel_manifest.json", {**vars(args), "assigned_jobs": jobs,
        "engine_sha256": engine_sha, "coefficients_sha256": coefficients_sha,
        "protocol_unchanged": True, "created_utc": datetime.now(timezone.utc).isoformat()})
    capacities = {gpu: args.workers_per_gpu for gpu in gpus}
    last_launch = {gpu: 0. for gpu in gpus}
    deadline_reached = False
    stop_dispatch = False
    dispatch_closed = False
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    env.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    log_handles = []
    def report(state, resources):
        fallback = [tag for tag in jobs if tag in failed or tag in interrupted or (dispatch_closed and tag in pending)]
        value = {"pid": os.getpid(), "manager_script": str(Path(__file__).resolve()), "state": state,
                 "role": args.role, "deadline": args.deadline, "assigned_jobs": jobs,
                 "workers_per_gpu": capacities, "gpu_resources": resources,
                 "workers": {tag: {"job": tag, "gpu": record["gpu"], "pid": record["pid"],
                                    "state": "running", "adopted": record["adopted"]} for tag, record in active.items()},
                 "completed": completed, "failed": failed, "interrupted_at_deadline": interrupted,
                 "pending": pending, "dispatch_closed": dispatch_closed, "remote_fallback_jobs": fallback,
                 "updated_utc": datetime.now(timezone.utc).isoformat()}
        atomic(args.state_dir / "status.json", value)
        atomic(args.state_dir / "live_handoff.json", value)
        lines = ["# Live local experiment handoff", "", f"State: {state}", f"Deadline: {args.deadline}",
                 f"Updated UTC: {value['updated_utc']}", "", "## Completed", "", *[f"- {tag}" for tag in completed],
                 "", "## Still running here", "", *[f"- {tag}" for tag in active],
                 "", "## Queued here", "", *[f"- {tag}" for tag in pending if tag not in fallback],
                 "", "## Released to remote", "", *[f"- {tag}" for tag in fallback],
                 "", "Incomplete runs must restart from step zero; all files are preserved."]
        (args.state_dir / "LIVE_HANDOFF.md").write_text("\n".join(lines) + "\n")
    while pending or active:
        for tag, record in list(active.items()):
            result = args.output / tag / "result.json"
            process = record["process"]
            rc = process.poll() if process is not None else None
            if result.exists():
                assert json.loads(result.read_text())["steps"] == 100
                completed.append(tag)
                del active[tag]
                print("DONE", tag, flush=True)
            elif (process is not None and rc is not None) or (process is None and not process_alive(record["pid"])):
                log = args.state_dir / f"{tag}.log"
                message = log.read_text(errors="replace")[-6000:] if log.exists() else ""
                failed[tag] = {"exit_code": rc, "gpu": record["gpu"], "oom": "out of memory" in message.lower()}
                if failed[tag]["oom"]:
                    capacities[record["gpu"]] = max(1, capacities[record["gpu"]] - 1)
                else:
                    stop_dispatch = True
                del active[tag]
                print("FAILED", tag, failed[tag], flush=True)
        now = time.time()
        if deadline is not None and now >= deadline:
            deadline_reached = True
            dispatch_closed = True
            for tag, record in list(active.items()):
                if validate_pid(record["pid"], tag):
                    os.kill(record["pid"], signal.SIGTERM)
                interrupted.append(tag)
            cutoff = time.time() + 10
            while any(process_alive(record["pid"]) for record in active.values()) and time.time() < cutoff:
                time.sleep(.5)
            for tag, record in active.items():
                if process_alive(record["pid"]) and validate_pid(record["pid"], tag):
                    os.kill(record["pid"], signal.SIGKILL)
            active.clear()
            break
        resources = gpu_status()
        can_finish = deadline is None or now + (args.estimated_run_minutes + args.finish_margin_minutes) * 60 < deadline
        dispatch_closed = not can_finish or stop_dispatch
        if not stop_dispatch and can_finish:
            for gpu in gpus:
                workers = [(tag, record) for tag, record in active.items() if record["gpu"] == gpu]
                if not pending or len(workers) >= capacities[gpu] or resources[gpu]["free_mib"] < args.min_free_mib:
                    continue
                if now - last_launch[gpu] < args.launch_interval_seconds:
                    continue
                tag = pending[0]
                entropy_present = any(value.startswith("component_zvp_") for value, _ in workers)
                if (entropy_present or tag.startswith("component_zvp_")) and len(workers) >= min(3, capacities[gpu]):
                    continue
                match = re.fullmatch(r"(.+)_s([123])_n100", tag)
                arm, seed = match.groups()
                assert hashlib.sha256(engine.read_bytes()).hexdigest() == engine_sha
                command = [sys.executable, "-u", str(engine), "--gpu", str(gpu), "--arm", arm,
                           "--seed", seed, "--steps", "100", "--eval-n", "60", "--model", model,
                           "--output", str(args.output.resolve()), "--min-free-mib", str(args.min_free_mib)]
                log = (args.state_dir / f"{tag}.log").open("x")
                log_handles.append(log)
                process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                active[tag] = {"gpu": gpu, "pid": process.pid, "adopted": False, "process": process}
                pending.pop(0)
                last_launch[gpu] = now
                print("START", tag, "gpu", gpu, "pid", process.pid, flush=True)
        report("running" if active else "deferred", resources)
        if not active and (not can_finish or stop_dispatch):
            break
        time.sleep(5)
    final = "deadline_reached" if deadline_reached else "complete" if len(completed) == len(jobs) else "remote_fallback"
    report(final, gpu_status())
    for log in log_handles:
        log.close()
    print("FINISHED", final, "completed", len(completed), "of", len(jobs), flush=True)


if __name__ == "__main__":
    main()
