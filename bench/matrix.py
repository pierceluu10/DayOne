"""Run many onboardings, each in its own process, skipping any already recorded under the same label.

    python -m bench.matrix --label matrix-off --memory off --db dayone_bench --workers 3
    python -m bench.matrix --label curve --memory on --db dayone_curve --workers 1 --repos click,gin,... --os-major
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from pymongo import MongoClient

from dayone.config import STATE_DIR, settings
from dayone.contracts import Machine, container_name

from bench.run_one import RUNS, load_repos

RUN_TIMEOUT_S = 3600.0  # one hire's whole onboarding; past this it counts as not onboarded
MACHINES = (Machine.UBUNTU, Machine.DEBIAN, Machine.ALPINE)
LOG_DIR = STATE_DIR / "bench"


def plan(repos: list[str], os_major: bool) -> list[tuple[str, Machine]]:
    if os_major:
        return [(repo, machine) for machine in MACHINES for repo in repos]
    return [(repo, machine) for repo in repos for machine in MACHINES]


def _run_job(args: argparse.Namespace, sequence: int, repo: str, machine: Machine) -> str:
    log = LOG_DIR / args.label / f"{sequence:03d}-{repo}-{machine.value}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "bench.run_one", "--repo", repo, "--machine", machine.value,
        "--memory", args.memory, "--db", args.db, "--label", args.label, "--sequence", str(sequence),
    ]  # fmt: skip
    with log.open("w") as out:
        process = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            process.wait(timeout=RUN_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            name = f"{args.label}-{repo}-{machine.value}"
            subprocess.run(["docker", "rm", "-f", container_name(name)], capture_output=True, check=False)
            _record_timeout(args, sequence, repo, machine)
            return f"{repo}/{machine.value}: timed out"
    return f"{repo}/{machine.value}: exit {process.returncode}"


def _record_timeout(args: argparse.Namespace, sequence: int, repo: str, machine: Machine) -> None:
    config = replace(settings(), mongodb_db=args.db)
    with MongoClient(config.mongodb_uri, tz_aware=True) as client:
        client[args.db][RUNS].insert_one(
            {
                "label": args.label, "sequence": sequence, "repo": repo, "machine": machine.value,
                "memory": args.memory, "ok": False, "error": f"timed out after {RUN_TIMEOUT_S:.0f}s",
                "elapsed_s": RUN_TIMEOUT_S, "timed_out": True, "finished_at": datetime.now(UTC),
            }
        )  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--memory", required=True, choices=["on", "off"])
    parser.add_argument("--db", required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--repos", default="", help="comma-separated names; default every repo in repos.json")
    parser.add_argument("--os-major", action="store_true", help="every repo on ubuntu, then debian, then alpine")
    args = parser.parse_args()

    repos = args.repos.split(",") if args.repos else list(load_repos())
    config = replace(settings(), mongodb_db=args.db)
    with MongoClient(config.mongodb_uri, tz_aware=True) as client:
        done = {(run["repo"], run["machine"]) for run in client[args.db][RUNS].find({"label": args.label})}
    jobs = [(sequence, repo, machine) for sequence, (repo, machine) in enumerate(plan(repos, args.os_major))]
    pending = [job for job in jobs if (job[1], job[2].value) not in done]
    print(f"{args.label}: {len(pending)} of {len(jobs)} runs to go", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for outcome in pool.map(lambda job: _run_job(args, *job), pending):
            print(f"{datetime.now(UTC):%H:%M:%S} {outcome}", flush=True)


if __name__ == "__main__":
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    main()
