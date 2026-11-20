"""Onboard one repository on one bare OS with the real engine, memory and Atlas, then record the outcome.

    python -m bench.run_one --repo click --machine alpine --memory on --db dayone_bench --label matrix-on
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import traceback
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from dayone.config import settings
from dayone.contracts import Hire, Machine, StepStatus, container_name
from dayone.engine.runner import Lease, Parts, Run
from dayone.llm.anthropic import AnthropicLLM
from dayone.llm.voyage import VoyageEmbedder
from dayone.memory.experience import ExperienceMemory
from dayone.store.atlas import AtlasStore
from dayone.store.events import AtlasEventBus
from dayone.terminal.docker import DockerExecutor
from dayone.terminal.fingerprint import fingerprint

from bench.steps import SETUP_TIMEOUT_S, Repo, build_steps

MANIFEST = Path(__file__).with_name("repos.json")
RUNS = "bench_runs"
# Bare images: nothing preinstalled, unlike the demo's pre-baked Ubuntu hire image.
IMAGES = {Machine.UBUNTU: "ubuntu:22.04", Machine.DEBIAN: "debian:12", Machine.ALPINE: "alpine:3.20"}
CONTAINER_LIMITS = ("--memory", "2000m", "--cpus", "3")
MAX_ATTEMPTS = 5  # reasoning attempts per failing step, the same budget with memory on and off
LLM_TIMEOUT_S = 240.0  # adaptive thinking on hard build failures outlasts the demo's 60 s
# USD per million tokens (input, output), from Anthropic's published price list.
PRICES = {"claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0), "claude-haiku-5-5": (0.1, 0.5)}
MEMORY_STATUSES = {StepStatus.FIXED_FROM_MEMORY}
REASONING_STATUSES = {StepStatus.FIXED_BY_REASONING}


class BenchRun(Run):
    def _close_loop(self) -> dict[str, object]:
        return {}  # benchmarks open no README issues


def load_repos() -> dict[str, Repo]:
    return {entry["name"]: Repo.from_manifest(entry) for entry in json.loads(MANIFEST.read_text())}


def _docker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=False)


def fresh_container(hire: Hire) -> None:
    _docker("rm", "-f", hire.container)
    created = _docker(
        "run", "-d", "--init", "--name", hire.container, "--hostname", hire.container,
        *CONTAINER_LIMITS, IMAGES[hire.machine], "tail", "-f", "/dev/null",
    )  # fmt: skip
    if created.returncode != 0:
        raise RuntimeError(f"cannot start {hire.container}: {created.stderr.strip()}")


def run(repo: Repo, machine: Machine, recall: bool, db: str, label: str, sequence: int) -> dict[str, object]:
    config = replace(settings(), mongodb_db=db, llm_timeout_s=LLM_TIMEOUT_S, exec_timeout_s=SETUP_TIMEOUT_S)
    name = f"{label}-{repo.name}-{machine.value}"
    hire = Hire(name=name, machine=machine, container=container_name(name), run_id=uuid.uuid4().hex)
    llm = AnthropicLLM(config)
    embedder = VoyageEmbedder(config)
    store = AtlasStore(config)
    started = time.monotonic()
    started_at = datetime.now(UTC)
    error = None
    ok = False
    try:
        store.reset_hire(name)
        fresh_container(hire)
        with Lease(store, name) as lease:
            store.upsert_hire(hire)
            events = AtlasEventBus(store.db, hire)
            try:
                # No Gitea and no browser: these steps never call them, and BenchRun opens no issues.
                memory = ExperienceMemory(
                    store.db, llm, embedder, events, None, config, recall=recall, max_attempts=MAX_ATTEMPTS
                )
                memory.ensure_indexes()
                parts = Parts(
                    events=events,
                    executor=DockerExecutor(hire, events, config.exec_timeout_s),
                    browser=None,
                    gitea=None,
                    memory=memory,
                    steps=build_steps(repo),
                    fingerprint=fingerprint,
                    exec_timeout_s=config.exec_timeout_s,
                )
                ok = BenchRun(store, parts, hire, lease).execute(resume=False)
            finally:
                events.close()
        steps = store.steps(name)
    except Exception as raised:  # noqa: BLE001 - any crash is a failed onboarding, recorded with its trace
        error = "".join(traceback.format_exception(raised))[-4000:]
        steps = store.steps(name)
    finally:
        _docker("rm", "-f", hire.container)
        embedder.close()
    elapsed = time.monotonic() - started
    price_in, price_out = PRICES.get(config.llm_model, (0.0, 0.0))
    record = {
        "label": label,
        "sequence": sequence,
        "repo": repo.name,
        "stack": repo.stack,
        "category": repo.category,
        "machine": machine.value,
        "memory": "on" if recall else "off",
        "ok": ok,
        "error": error,
        "elapsed_s": round(elapsed, 1),
        "llm_model": config.llm_model,
        "llm_calls": llm.calls,
        "input_tokens": llm.input_tokens,
        "output_tokens": llm.output_tokens,
        "cost_usd": round((llm.input_tokens * price_in + llm.output_tokens * price_out) / 1e6, 4),
        "fixed_from_memory": sum(step.status in MEMORY_STATUSES for step in steps.values()),
        "fixed_by_reasoning": sum(step.status in REASONING_STATUSES for step in steps.values()),
        "steps": {
            step_id: {"status": step.status.value, "evidence": step.evidence[:300], "source": step.fix_source}
            for step_id, step in steps.items()
        },
        "started_at": started_at,
        "finished_at": datetime.now(UTC),
    }
    try:
        store.db[RUNS].insert_one(dict(record))
    finally:
        store.close()
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--machine", required=True, choices=[machine.value for machine in Machine])
    parser.add_argument("--memory", required=True, choices=["on", "off"])
    parser.add_argument("--db", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--sequence", type=int, default=0)
    args = parser.parse_args()
    record = run(load_repos()[args.repo], Machine(args.machine), args.memory == "on", args.db, args.label, args.sequence)
    print(json.dumps({key: value for key, value in record.items() if key != "steps"}, default=str))
    print(json.dumps(record["steps"], default=str))


if __name__ == "__main__":
    main()
