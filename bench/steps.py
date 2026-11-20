"""The onboarding workflow for any repository in `repos.json`: base tools, clone, setup, tests.

Each run() is the naive action a contributor guide implies; each check() verifies the goal by another route,
so whatever the guide leaves out surfaces as a failing check whose error text memory then fixes.
Step ids for setup and tests carry the repository's name: Atlas Vector Search filters on step id, so a setup fix
is retrieved for the same project on any OS, while base-tool fixes are shared by every project on that OS.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass

from dayone.contracts import REPO_DIR, CheckResult, ExecResult, RunResult, Step, StepContext, StepKind
from dayone.terminal.docker import error_text

QUICK_TIMEOUT_S = 120.0
SETUP_TIMEOUT_S = 1200.0
TEST_TIMEOUT_S = 900.0
RUN_OUTPUT_CHARS = 3000  # the setup commands' own output, appended to the ready check's error
BASE_CHECK = "git --version && curl --version && curl -fsSI https://github.com >/dev/null"
# Lockfiles legitimately change when dependencies resolve; any other tracked change means the agent edited the project.
LOCKFILES = r"(\.lock|lock\.json|go\.sum|Cargo\.lock|yarn\.lock)$"
TRACKED_CHANGES = f"git diff --name-only | grep -vE '{LOCKFILES}' || true"


@dataclass(frozen=True)
class Repo:
    name: str
    url: str
    ref: str
    stack: str
    category: str
    setup: tuple[str, ...]
    ready: str
    test: str
    module: str  # Python only: the package that must import from the checkout, not from a package index

    @classmethod
    def from_manifest(cls, entry: dict[str, object]) -> Repo:
        return cls(
            name=str(entry["name"]),
            url=str(entry["url"]),
            ref=str(entry["ref"]),
            stack=str(entry["stack"]),
            category=str(entry["category"]),
            setup=tuple(str(command) for command in entry["setup"]),  # type: ignore[union-attr]
            ready=str(entry["ready"]),
            test=str(entry["test"]),
            module=str(entry.get("module", "")),
        )


def _exec(ctx: StepContext, command: str, timeout_s: float, workdir: str | None = None) -> ExecResult:
    return ctx.executor.exec(command, timeout_s, workdir)


def _failed(result: ExecResult, workdir: str | None, evidence: str, extra: str = "") -> CheckResult:
    return CheckResult(ok=False, evidence=evidence, error_text=error_text(result, workdir) + extra)


def base_check(ctx: StepContext) -> CheckResult:
    result = _exec(ctx, BASE_CHECK, QUICK_TIMEOUT_S)
    if not result.ok:
        return _failed(result, None, "git, curl or CA certificates missing")
    return CheckResult(ok=True, evidence=result.output.splitlines()[0])


def nothing(ctx: StepContext) -> RunResult:
    return RunResult(ok=True, output="")


def build_steps(repo: Repo) -> list[Step]:
    clone = (
        f"mkdir -p {REPO_DIR} && cd {REPO_DIR} && git init -q && "
        f"(git remote add origin {shlex.quote(repo.url)} || true) && "
        f"git fetch -q --depth 1 origin {repo.ref} && git checkout -q FETCH_HEAD"
    )
    last_setup: list[str] = [""]  # the setup run's output, which the ready check's error must carry

    def clone_check(ctx: StepContext) -> CheckResult:
        result = _exec(ctx, f"git -C {REPO_DIR} rev-parse HEAD", QUICK_TIMEOUT_S)
        head = result.output.strip()
        if result.ok and head == repo.ref:
            return CheckResult(ok=True, evidence=f"HEAD {head[:7]} = pinned {repo.ref[:7]}")
        return _failed(result, None, f"HEAD is not {repo.ref[:7]}")

    def clone_run(ctx: StepContext) -> RunResult:
        result = _exec(ctx, clone, SETUP_TIMEOUT_S)
        return RunResult(ok=result.ok, output=result.output, commands=(clone,))

    def setup_run(ctx: StepContext) -> RunResult:
        outputs = []
        ok = True
        for command in repo.setup:
            result = _exec(ctx, command, SETUP_TIMEOUT_S, REPO_DIR)
            outputs.append(f"$ {command}\n{result.output}")
            ok = result.ok
            if not ok:
                break
        last_setup[0] = "\n".join(outputs)
        return RunResult(ok=ok, output=last_setup[0], commands=repo.setup)

    def setup_check(ctx: StepContext) -> CheckResult:
        result = _exec(ctx, repo.ready, SETUP_TIMEOUT_S, REPO_DIR)
        if result.ok:
            return CheckResult(ok=True, evidence=f"ready: {repo.ready}")
        extra = f"\nThe documented setup commands printed:\n{last_setup[0][-RUN_OUTPUT_CHARS:]}" if last_setup[0] else ""
        return _failed(result, REPO_DIR, "dependencies not ready", extra)

    def tests_check(ctx: StepContext) -> CheckResult:
        edited = _exec(ctx, TRACKED_CHANGES, QUICK_TIMEOUT_S, REPO_DIR).output.split()
        if edited:
            return CheckResult(
                ok=False,
                evidence=f"tracked project files were edited: {', '.join(edited[:5])}",
                error_text=f"The project's tracked files must stay unmodified; these were changed: {edited}. "
                f"Restore them with `git checkout -- <file>` and fix the environment instead.",
            )
        if repo.module:
            # Run from the checkout, as the tests import it: a released copy from PyPI would pass the suite
            # without the project ever being built.
            probe = f'.venv/bin/python -c "import os, {repo.module}; print(os.path.realpath({repo.module}.__file__))"'
            located = _exec(ctx, probe, QUICK_TIMEOUT_S, REPO_DIR)
            if not located.ok or not located.output.strip().startswith(REPO_DIR):
                return CheckResult(
                    ok=False,
                    evidence=f"{repo.module} is not imported from the checkout",
                    error_text=error_text(located, REPO_DIR)
                    + f"\n{repo.module} must be built and installed from this checkout (e.g. pip install -e), "
                    "not a released copy from a package index.",
                )
        result = _exec(ctx, repo.test, TEST_TIMEOUT_S, REPO_DIR)
        if not result.ok:
            return _failed(result, REPO_DIR, f"tests exit {result.exit_code}")
        summary = next((line for line in reversed(result.output.splitlines()) if line.strip()), "")
        return CheckResult(ok=True, evidence=f"exit 0: {summary[:160]}")

    return [
        Step("base", StepKind.TERMINAL, "Install base tools", base_check, nothing),
        Step("clone", StepKind.TERMINAL, f"Clone {repo.name}", clone_check, clone_run),
        Step(f"setup:{repo.name}", StepKind.TERMINAL, "Install dependencies", setup_check, setup_run),
        Step(f"tests:{repo.name}", StepKind.TERMINAL, "Run the test suite", tests_check, nothing),
    ]
