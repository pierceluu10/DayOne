"""Experience memory on Atlas: exact fingerprint, then $vectorSearch (adapting apt -> apk), then bounded LLM reasoning."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from typing import Any

from bson import ObjectId
from pymongo import DESCENDING
from pymongo.database import Database
from pymongo.operations import SearchIndexModel

from dayone.config import Settings
from dayone.contracts import (
    LLM,
    REPO,
    REPO_DIR,
    CheckResult,
    Embedder,
    EventBus,
    EventType,
    Failure,
    FixProposal,
    FixSource,
    Gitea,
    LocatorHint,
    MachineFacts,
    Resolution,
    StepKind,
)

EXPERIENCES = "experiences"
VECTOR_INDEX = "experience_embedding"
VERIFIED = "verified"
FAILED = "failed"
DOCUMENT = "document"
QUERY = "query"

INDEX_POLL_S = 5.0
INDEX_READY = "READY"
INDEX_WAIT_S = 300.0
NUM_CANDIDATES = 100
HINT_COUNT = 3
NEIGHBOUR_COUNT = 10  # enough to find a same-machine hit ranked below other machines' copies of the fix
SEMANTIC_MIN_SIMILARITY = 0.8  # Atlas cosine score; Phase 0 saw 0.847 for the right fix, 0.73 for the next one
MAX_REASONING_ATTEMPTS = 3  # fixes actually tried
MAX_REJECTED_REPLIES = 2  # unparseable replies re-asked on top of those, so a bad reply never costs a try
MAX_TOKENS = 8192  # adaptive thinking counts against it, and a cut-off reply is rejected and wastes an attempt
ERROR_TAIL_CHARS = 4000  # pip/apt failures put the cause at the end
FEEDBACK_CHARS = 3000  # all fix commands' output fed back to the next attempt, split evenly
FIX_OUTPUT = "fix_output"  # CheckResult.detail key: [{"command", "exit_code", "output"}] from try_fix
ISSUE_ERROR_LINES = 15
ISSUE_TITLE_COMMAND_CHARS = 120
README_MIN_HIRES = 2

JSON_ESCAPE = re.compile(r'\\(["\\/bfnrtu])|\\')  # a valid escape pair, or a lone backslash
APT_INSTALL = re.compile(r"^(apt-get|apt)\s+install\b")
ASSUME_YES = re.compile(r"\s(-y|--yes|--assume-yes)\b")
SUDO = "sudo "
APT_NONINTERACTIVE = "DEBIAN_FRONTEND=noninteractive "
NON_ABORTING = "|| true"

FIX_RULES = f"""Commands run as root with `sh -lc` in the new hire's throwaway container, in {REPO_DIR} when it exists.
Use the machine's own package manager. Non-interactive only (-y, --no-cache), no sudo, one command per string,
do not chain with && or ;. Commands from earlier attempts already ran; only add what is still needed.
Each command runs in a fresh login shell: a toolchain installed outside the system PATH (rustup, nvm, a Go or Ruby
tarball) must also be put on PATH for later shells, e.g. by linking its binaries into /usr/local/bin.
Services a step needs (databases, caches) must be started in the background so the command returns.
For a Python project, its Python is the venv at {REPO_DIR}/.venv: use .venv/bin/python and .venv/bin/pip for
anything in the project, never the system python or pip.
Never edit, skip, deselect or delete the project's tests or source to make a check pass, and build the project
itself from the checkout (e.g. an editable install), never a released copy from a package index."""
STRICT_JSON = "Reply with strict JSON and nothing else: "

REASON_SYSTEM = f"""You fix failed setup steps on a new hire's development machine.
The team's repository is cloned at {REPO_DIR} (once the clone step has passed): its files, scripts and docs often
name the exact setup command. You may read them first with read-only `inspect` commands (ls, cat, grep); their
output, with every command's exit code, comes back to you if the step still fails. Every attempt must also try a fix
in `commands`, and `commands` must hold only the fix, never inspection.
{FIX_RULES}
{STRICT_JSON}{{"inspect": ["..."], "commands": ["..."], "rationale": "one sentence"}}"""
ADAPT_SYSTEM = (
    "You adapt a setup fix that worked on one machine to a different machine (for example apt to apk, or Ubuntu to Debian), "
    f"keeping its intent and translating package names.\n{FIX_RULES}\n"
    f'{STRICT_JSON}{{"commands": ["..."], "rationale": "one sentence"}}'
)


class LLMResponseError(ValueError):
    """The model's reply was not the strict JSON fix we asked for."""


class IndexNotReady(TimeoutError):
    pass


@dataclass
class _Progress:
    initial: CheckResult
    attempts: int = 0
    llm_calls: int = 0
    last_check: CheckResult | None = None

    def resolution(self, proposal: FixProposal | None) -> Resolution:
        return Resolution(
            fixed=proposal is not None,
            proposal=proposal,
            attempts=self.attempts,
            llm_calls=self.llm_calls,
            last_check=self.last_check or self.initial,
        )


def _now() -> datetime:
    return datetime.now(UTC)


def _tail(text: str) -> str:
    return text[-ERROR_TAIL_CHARS:]


def _memory_text(error_text: str, context_summary: str) -> str:
    return f"{_tail(error_text)}\n{context_summary}"


def _facts_from(context: dict[str, Any]) -> MachineFacts:
    return MachineFacts(
        os_family=context["os_family"],
        os_version=context["os_version"],
        package_manager=context["package_manager"],
        tools=context.get("tools", {}),
    )


def _machine(facts: MachineFacts) -> str:
    """What a fix depends on: "ubuntu 22.04, apt". Tool versions are left out, the fix usually changes them."""
    return f"{facts.os_family} {facts.os_version}, {facts.package_manager}"


def _proposal_data(proposal: FixProposal) -> dict[str, Any]:
    return {**asdict(proposal), "commands": list(proposal.commands)}


def guard_commands(commands: list[str]) -> tuple[str, ...]:
    """Make model-written commands runnable unattended: one per line, no sudo, apt installs non-interactive."""
    guarded = []
    for raw in commands:
        for line in raw.splitlines():
            command = line.strip().removeprefix(SUDO).strip()
            if not command or command.startswith("#"):
                continue
            if APT_INSTALL.match(command):
                if not ASSUME_YES.search(command):
                    command = command.replace(" install", " install -y", 1)
                command = APT_NONINTERACTIVE + command
            guarded.append(command)
    return tuple(guarded)


def _strings(body: dict[str, Any], key: str) -> list[str]:
    value = body.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise LLMResponseError(f"`{key}` must be a list of strings")
    return value


def parse_fix(reply: str) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    """(inspect, commands, rationale) from the model's strict-JSON reply; `commands` is the fix and never empty."""
    start, end = reply.find("{"), reply.rfind("}")
    if start < 0 or end < start:
        raise LLMResponseError(f"no JSON object in reply: {reply[:200]!r}")
    text = reply[start : end + 1]
    try:
        body = json.loads(text)
    except json.JSONDecodeError as error:
        # Models write shell regexes like `grep 'a\|b'` without doubling the backslash; keep it literal and retry.
        repaired = JSON_ESCAPE.sub(lambda match: match.group(0) if match.group(1) else "\\\\", text)
        try:
            body = json.loads(repaired)
        except json.JSONDecodeError:
            raise LLMResponseError(f"invalid JSON: {error}") from error
    if "commands" not in body:
        raise LLMResponseError("`commands` is missing")
    commands = guard_commands(_strings(body, "commands"))
    if not commands:
        raise LLMResponseError("no runnable commands in the reply")
    # try_fix stops at the first failing command, and a grep with no match exits 1: inspection must never abort the fix.
    inspect = tuple(
        command if command.endswith(NON_ABORTING) else f"{command} {NON_ABORTING}"
        for command in guard_commands(_strings(body, "inspect"))
    )
    return inspect, commands, str(body.get("rationale", ""))


def succeeded(commands: tuple[str, ...], check: CheckResult) -> list[str]:
    """The fix commands that exited 0, per the engine's `detail[FIX_OUTPUT]`."""
    results = check.detail.get(FIX_OUTPUT)
    if results is None:
        # shortcut: a try_fix without FIX_OUTPUT gives no exit codes, so keep every command; all engines send it now.
        return list(commands)
    exited_ok = {result["command"] for result in results if result["exit_code"] == 0}
    return [command for command in commands if command in exited_ok]


def command_feedback(check: CheckResult) -> str:
    """What the fix commands printed, from the engine's `detail[FIX_OUTPUT]`, capped so the prompt stays small."""
    results = check.detail.get(FIX_OUTPUT, [])
    if not results:
        return ""
    budget = FEEDBACK_CHARS // len(results)
    return "\n".join(
        f"$ {result['command']}  (exit {result['exit_code']})\n{result['output'][-budget:].rstrip()}" for result in results
    )


class ExperienceMemory:
    def __init__(
        self,
        db: Database,
        llm: LLM,
        embedder: Embedder,
        events: EventBus,
        gitea: Gitea,
        settings: Settings,
        recall: bool = True,
        max_attempts: int = MAX_REASONING_ATTEMPTS,
    ) -> None:
        """`recall=False` never reads memory (no exact, semantic or hint lookups) but still saves what it learns:
        the benchmarks' memory-off baseline."""
        self._recall = recall
        self._max_attempts = max_attempts
        self._experiences = db[EXPERIENCES]
        self._llm = llm
        self._embedder = embedder
        self._events = events
        self._gitea = gitea
        self._dimensions = settings.embedding_dimensions

    # --- indexes ----------------------------------------------------------------

    def ensure_indexes(self) -> None:
        self._experiences.create_index("fingerprint")  # also creates the collection the search index needs
        definition = {
            "fields": [
                {"type": "vector", "path": "embedding", "numDimensions": self._dimensions, "similarity": "cosine"},
                {"type": "filter", "path": "outcome"},
                {"type": "filter", "path": "kind"},
                {"type": "filter", "path": "context.step_id"},
            ]
        }
        existing = list(self._experiences.list_search_indexes(VECTOR_INDEX))
        if not existing:
            self._experiences.create_search_index(
                SearchIndexModel(definition=definition, name=VECTOR_INDEX, type="vectorSearch")
            )
        elif existing[0]["latestDefinition"] != definition:
            self._experiences.update_search_index(VECTOR_INDEX, definition)
        deadline = time.monotonic() + INDEX_WAIT_S
        while not self._index_ready(definition):
            if time.monotonic() > deadline:
                raise IndexNotReady(f"{VECTOR_INDEX} not ready after {INDEX_WAIT_S:.0f}s")
            time.sleep(INDEX_POLL_S)

    def _index_ready(self, definition: dict[str, Any]) -> bool:
        # After an update the old version stays queryable while the new one builds; filters on new fields fail until
        # the new version is READY, so wait for status as well as queryable.
        indexes = list(self._experiences.list_search_indexes(VECTOR_INDEX))
        return bool(indexes) and all(
            index.get("queryable") and index.get("status") == INDEX_READY and index["latestDefinition"] == definition
            for index in indexes
        )

    # --- terminal fixes -----------------------------------------------------------

    def resolve(self, failure: Failure, try_fix: Callable[[FixProposal], CheckResult]) -> Resolution:
        progress = _Progress(initial=CheckResult(ok=False, evidence="step check failed", error_text=failure.error_text))
        if not self._recall:
            return self._reason(failure, try_fix, [], progress)

        exact = self._exact(failure)
        if exact:
            proposal = FixProposal(
                commands=tuple(exact["fix_commands"]),
                source=FixSource.EXACT_MEMORY,
                rationale=exact.get("rationale", ""),
                experience_id=str(exact["_id"]),
                source_hire=exact["hire"],
            )
            self._events.emit(EventType.MEMORY_HIT, failure.step_id, **_proposal_data(proposal), uses=exact["uses"])
            if self._try(failure, proposal, try_fix, progress):
                self._record_reuse(exact["_id"], failure.hire)
                return progress.resolution(proposal)

        neighbours = self._nearest(failure, exclude=exact["_id"] if exact else None)
        semantic = self._pick(failure, neighbours)
        if semantic:
            proposal = self._from_semantic(failure, semantic, progress)
            if proposal:
                self._events.emit(EventType.MEMORY_HIT, failure.step_id, **_proposal_data(proposal))
                if self._try(failure, proposal, try_fix, progress):
                    if proposal.adapted_from:
                        saved = self._save(
                            failure, proposal.commands, progress, proposal, adapted_from_experience=semantic["_id"]
                        )
                        proposal = replace(proposal, experience_id=str(saved))
                    else:
                        self._record_reuse(semantic["_id"], failure.hire)
                    return progress.resolution(proposal)

        if not exact and not semantic:
            self._events.emit(EventType.MEMORY_MISS, failure.step_id, fingerprint=failure.fingerprint)
        return self._reason(failure, try_fix, neighbours[:HINT_COUNT], progress)

    def _exact(self, failure: Failure) -> dict[str, Any] | None:
        return self._experiences.find_one(
            {
                "fingerprint": failure.fingerprint,
                "outcome": VERIFIED,
                "kind": failure.kind,
                # Same machine only: ubuntu's deadsnakes PPA fix fails on debian although both use apt.
                "context.os_family": failure.facts.os_family,
                "context.os_version": failure.facts.os_version,
                "context.package_manager": failure.facts.package_manager,
            },
            sort=[("uses", DESCENDING)],
        )

    def _nearest(self, failure: Failure, exclude: ObjectId | None) -> list[dict[str, Any]]:
        [vector] = self._embedder.embed([_memory_text(failure.error_text, failure.context_summary())], QUERY)
        pipeline: list[dict[str, Any]] = [
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX,
                    "path": "embedding",
                    "queryVector": vector,
                    "numCandidates": NUM_CANDIDATES,
                    "limit": NEIGHBOUR_COUNT + 1,  # room for the excluded exact match
                    # Same step only: a t_runtime fix scored 0.822 against a t_system_package error and wasted a try.
                    "filter": {"outcome": VERIFIED, "kind": failure.kind, "context.step_id": failure.step_id},
                }
            },
            {"$set": {"score": {"$meta": "vectorSearchScore"}}},
            {"$project": {"embedding": 0}},
        ]
        if exclude:
            pipeline.append({"$match": {"_id": {"$ne": exclude}}})
        pipeline.append({"$limit": NEIGHBOUR_COUNT})
        return list(self._experiences.aggregate(pipeline))

    def _pick(self, failure: Failure, neighbours: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Best hit above the threshold, preferring this machine's own: reuse it as-is and credit its hires, rather
        than adapting another machine's adaptation of it."""
        above = [hit for hit in neighbours if hit["score"] >= SEMANTIC_MIN_SIMILARITY]
        machine = _machine(failure.facts)
        same = [hit for hit in above if _machine(_facts_from(hit["context"])) == machine]
        return (same or above or [None])[0]

    def _from_semantic(self, failure: Failure, hit: dict[str, Any], progress: _Progress) -> FixProposal | None:
        base = FixProposal(
            commands=tuple(hit["fix_commands"]),
            source=FixSource.SEMANTIC_MEMORY,
            rationale=hit.get("rationale", ""),
            experience_id=str(hit["_id"]),
            similarity=hit["score"],
            source_hire=hit["hire"],
        )
        original = _facts_from(hit["context"])
        if _machine(original) == _machine(failure.facts):
            return base
        prompt = (
            f"This fix was verified on {original.summary()} for this error:\n{_tail(hit['error_text'])}\n\n"
            "Commands that fixed it there:\n" + "\n".join(hit["fix_commands"]) + "\n\n"
            f"Now machine {failure.facts.summary()} fails step {failure.step_id} with:\n{_tail(failure.error_text)}\n\n"
            f"Rewrite the commands for {_machine(failure.facts)}; anything specific to {original.os_family} "
            "(its repositories, PPAs, package names) must become this machine's equivalent."
        )
        progress.llm_calls += 1
        try:
            _, commands, rationale = parse_fix(self._llm.complete(ADAPT_SYSTEM, prompt, MAX_TOKENS))
        except LLMResponseError:
            return None
        return replace(base, commands=commands, rationale=rationale, adapted_from=_machine(original))

    def _reason(
        self,
        failure: Failure,
        try_fix: Callable[[FixProposal], CheckResult],
        hints: list[dict[str, Any]],
        progress: _Progress,
    ) -> Resolution:
        self._events.emit(EventType.REASONING_STARTED, failure.step_id, hints=[str(hint["_id"]) for hint in hints])
        hint_text = "\n\n".join(
            f"- on {_facts_from(hint['context']).summary()} (similarity {hint['score']:.2f}), error:\n"
            f"{_tail(hint['error_text'])}\n  fixed by: " + " ; ".join(hint["fix_commands"])
            for hint in hints
        )
        history: list[str] = []
        # The saved fix is this chain only: a memory proposal that already failed is not part of what fixed it, and a
        # command that failed or never ran would abort the next hire's replay (try_fix stops at the first failure).
        chain: list[str] = []
        error = failure.error_text
        tried = rejected = 0
        while tried < self._max_attempts:
            prompt = (
                f"Machine: {failure.facts.summary()}\nStep {failure.step_id} fails with:\n{_tail(error)}\n\n"
                + (f"Past verified fixes for similar errors (hints, may not apply):\n{hint_text}\n\n" if hints else "")
                + ("Earlier attempts in this session:\n" + "\n\n".join(history) + "\n\n" if history else "")
                + "Propose the commands that make this step pass."
            )
            progress.llm_calls += 1
            try:
                inspect, commands, rationale = parse_fix(self._llm.complete(REASON_SYSTEM, prompt, MAX_TOKENS))
            except LLMResponseError as reply_error:
                rejected += 1
                if rejected > MAX_REJECTED_REPLIES:
                    break
                history.append(f"Your reply was rejected, nothing ran: {reply_error}")
                continue
            tried += 1
            proposal = FixProposal(commands=inspect + commands, source=FixSource.REASONING, rationale=rationale)
            passed = self._try(failure, proposal, try_fix, progress)
            chain.extend(succeeded(commands, progress.last_check))
            if passed:
                saved = self._save(failure, tuple(chain), progress, proposal)
                return progress.resolution(replace(proposal, experience_id=str(saved)))
            check = progress.last_check
            error = check.error_text or check.evidence
            self._save_failed(failure, proposal.commands, check)
            output = command_feedback(check)
            history.append(
                "Ran:\n"
                + "\n".join(proposal.commands)
                + (f"\nCommand output:\n{output}" if output else "")
                + f"\nThe step's check still fails with:\n{_tail(error)}"
            )
        return progress.resolution(None)

    def _try(
        self,
        failure: Failure,
        proposal: FixProposal,
        try_fix: Callable[[FixProposal], CheckResult],
        progress: _Progress,
    ) -> bool:
        check = try_fix(proposal)
        progress.attempts += 1
        progress.last_check = check
        self._events.emit(
            EventType.FIX_ATTEMPT,
            failure.step_id,
            source=proposal.source,
            commands=list(proposal.commands),
            ok=check.ok,
            evidence=check.evidence,
        )
        return check.ok

    # --- persistence ----------------------------------------------------------------

    def _context(self, failure: Failure) -> dict[str, Any]:
        return {"step_id": failure.step_id, **asdict(failure.facts)}

    def _save(
        self,
        failure: Failure,
        fix_commands: tuple[str, ...],
        progress: _Progress,
        proposal: FixProposal,
        adapted_from_experience: ObjectId | None = None,
    ) -> ObjectId:
        [vector] = self._embedder.embed([_memory_text(failure.error_text, failure.context_summary())], DOCUMENT)
        document = {
            "kind": failure.kind,
            "fingerprint": failure.fingerprint,
            "error_text": failure.error_text,
            "context": self._context(failure),
            "fix_commands": list(fix_commands),
            "rationale": proposal.rationale,
            "source": proposal.source,
            "adapted_from": proposal.adapted_from,
            "adapted_from_experience": adapted_from_experience,
            "outcome": VERIFIED,
            "verified_by_check": progress.last_check.evidence,
            "hire": failure.hire,
            "hires": [failure.hire],
            "uses": 1,
            "embedding": vector,
            "created_at": _now(),
        }
        experience_id = self._experiences.insert_one(document).inserted_id
        self._events.emit(
            EventType.EXPERIENCE_SAVED,
            failure.step_id,
            experience_id=str(experience_id),
            fingerprint=failure.fingerprint,
            commands=document["fix_commands"],
            source=proposal.source,
        )
        return experience_id

    def _save_failed(self, failure: Failure, commands: tuple[str, ...], check: CheckResult) -> None:
        # No embedding: failed attempts are kept for the record and must never come back as a fix.
        self._experiences.insert_one(
            {
                "kind": failure.kind,
                "fingerprint": failure.fingerprint,
                "error_text": failure.error_text,
                "context": self._context(failure),
                "fix_commands": list(commands),
                "outcome": FAILED,
                "result_error": check.error_text or check.evidence,
                "hire": failure.hire,
                "created_at": _now(),
            }
        )

    def _record_reuse(self, experience_id: ObjectId, hire: str) -> None:
        self._experiences.update_one(
            {"_id": experience_id}, {"$inc": {"uses": 1}, "$addToSet": {"hires": hire}, "$set": {"last_used_at": _now()}}
        )

    # --- browser locators -------------------------------------------------------------

    def find_locator(self, goal: str, page_summary: str) -> LocatorHint | None:
        # Exact goal only, no vector search: every goal on one page shares its page summary and near-identical
        # wording ("sign-in page: username field" vs "...: remember this device checkbox"), so a semantic hit hands
        # one goal another goal's element. A same-goal hit is already found here, newest first, whatever the layout;
        # a moved element is caught when the locator fails and the browser reasons over the page (page_summary).
        document = self._experiences.find_one(
            {"kind": StepKind.BROWSER, "outcome": VERIFIED, "goal": goal}, sort=[("updated_at", DESCENDING)]
        )
        if document is None:
            return None
        bbox = document.get("bbox")
        return LocatorHint(
            goal=document["goal"],
            locator=document["locator"],
            bbox=tuple(bbox) if bbox else None,
            page_summary=document["page_summary"],
            source_hire=document["hire"],
            similarity=None,
        )

    def save_locator(self, hint: LocatorHint, hire: str) -> None:
        self._experiences.update_one(
            {"kind": StepKind.BROWSER, "outcome": VERIFIED, "goal": hint.goal, "locator": hint.locator},
            {
                "$set": {
                    "bbox": list(hint.bbox) if hint.bbox else None,
                    "page_summary": hint.page_summary,
                    "updated_at": _now(),
                },
                "$inc": {"uses": 1},
                "$addToSet": {"hires": hire},
                "$setOnInsert": {"hire": hire, "created_at": _now()},
            },
            upsert=True,
        )

    # --- closing the loop ---------------------------------------------------------------

    def close_loop(self) -> list[str]:
        repeated = self._experiences.aggregate(
            [
                {"$match": {"kind": StepKind.TERMINAL, "outcome": VERIFIED}},
                {"$match": {"$expr": {"$gte": [{"$size": "$hires"}, README_MIN_HIRES]}}},
                {"$project": {"embedding": 0}},
                {"$sort": {"created_at": 1}},
            ]
        )
        urls = []
        for experience in repeated:
            title, body = self._readme_issue(experience)
            issue = self._gitea.find_issue(REPO, title)
            if issue is None:
                issue = self._gitea.create_issue(REPO, title, body)
                self._events.emit(
                    EventType.README_ISSUE_OPENED,
                    experience["context"]["step_id"],
                    url=issue["html_url"],
                    title=title,
                    hires=experience["hires"],
                    experience_id=str(experience["_id"]),
                )
            urls.append(issue["html_url"])
        return urls

    def _readme_issue(self, experience: dict[str, Any]) -> tuple[str, str]:
        context = experience["context"]
        facts = _facts_from(context)
        commands = experience["fix_commands"]
        summary = "; ".join(commands)
        if len(summary) > ISSUE_TITLE_COMMAND_CHARS:
            summary = summary[: ISSUE_TITLE_COMMAND_CHARS - 1] + "…"
        title = f"README: {context['step_id']} on {facts.os_family} needs `{summary}`"
        error_lines = "\n".join(experience["error_text"].strip().splitlines()[-ISSUE_ERROR_LINES:])
        command_lines = "\n".join(commands)
        body = (
            f"{len(experience['hires'])} new hires ({', '.join(experience['hires'])}) on {facts.summary()} "
            f"hit this during `{context['step_id']}`, and the README does not mention it:\n\n"
            f"```\n{error_lines}\n```\n\n"
            f"Day One fixed it with these commands, verified by the step's check (`{experience['verified_by_check']}`):\n\n"
            f"```sh\n{command_lines}\n```\n\n"
            "Proposed change: add these commands to the README's setup section for this platform.\n\n"
            f"Opened by Day One from experience `{experience['_id']}`, applied {experience['uses']} times."
        )
        return title, body
