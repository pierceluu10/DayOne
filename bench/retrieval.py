"""Retrieval eval: does memory hand a failing hire a fix that addresses its error?

Queries are real failures the benchmark hires hit (every experience document carries its error text and machine).
Candidates are pooled from every retrieval system under test; Claude labels each (error, candidate fix) pair once,
and the labels are cached in Atlas so systems and thresholds are compared on identical judgments.

    python -m bench.retrieval --db dayone_bench --out bench/results/retrieval.json
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
from pymongo import MongoClient

from dayone.config import settings
from dayone.llm.anthropic import AnthropicLLM
from dayone.llm.voyage import VoyageEmbedder
from dayone.memory.experience import EXPERIENCES, NUM_CANDIDATES, QUERY, VECTOR_INDEX, VERIFIED, _facts_from, _memory_text, _tail

LABELS = "retrieval_labels"
POOL_DEPTH = 5  # candidates per system per query sent to the judge
RERANK_MODEL = "rerank-2.5"
THRESHOLDS = [round(0.50 + 0.025 * step, 3) for step in range(19)]  # 0.50 .. 0.95
PRODUCTION_THRESHOLD = 0.8
JUDGE_MAX_TOKENS = 4096
JUDGE_WORKERS = 6
JUDGE_SYSTEM = """You label search results for an onboarding agent's fix memory.
You get an error a new hire's machine hit, and a fix that an earlier hire verified for an error of their own.
Answer whether applying this fix (adapted to the new machine's package manager if needed) addresses the root cause
of the new error. Reply with strict JSON and nothing else: {"relevant": true|false, "reason": "one sentence"}"""


def _client(db: str) -> Any:
    config = replace(settings(), mongodb_db=db)
    return MongoClient(config.mongodb_uri, tz_aware=True)[db]


def _queries(database: Any) -> list[dict[str, Any]]:
    """One query per distinct failure (hire, step, fingerprint); clone and base-tool errors are trivial and left out."""
    seen: set[tuple[str, str, str]] = set()
    queries = []
    for doc in database[EXPERIENCES].find(
        {"kind": "terminal", "context.step_id": {"$regex": "^(setup|tests):"}}, {"embedding": 0}
    ).sort("created_at", 1):
        key = (doc["hire"], doc["context"]["step_id"], doc["fingerprint"])
        if key in seen:
            continue
        seen.add(key)
        queries.append(doc)
    return queries


def _vector(database: Any, vector: list[float], step_id: str | None, exclude_hire: str) -> list[dict[str, Any]]:
    search_filter: dict[str, Any] = {"outcome": VERIFIED, "kind": "terminal"}
    if step_id:
        search_filter["context.step_id"] = step_id
    pipeline = [
        {"$vectorSearch": {"index": VECTOR_INDEX, "path": "embedding", "queryVector": vector,
                           "numCandidates": NUM_CANDIDATES, "limit": 20, "filter": search_filter}},
        {"$set": {"score": {"$meta": "vectorSearchScore"}}},
        {"$project": {"embedding": 0}},
        # A hire's own saved fix was made from this very failure: retrieving it is not memory, it is the answer key.
        {"$match": {"hire": {"$ne": exclude_hire}}},
        {"$limit": 10},
    ]  # fmt: skip
    return list(database[EXPERIENCES].aggregate(pipeline))


def _exact(database: Any, query: dict[str, Any], same_machine: bool) -> list[dict[str, Any]]:
    match: dict[str, Any] = {"fingerprint": query["fingerprint"], "outcome": VERIFIED, "kind": "terminal",
                             "hire": {"$ne": query["hire"]}}  # fmt: skip
    if same_machine:
        for field in ("os_family", "os_version", "package_manager"):
            match[f"context.{field}"] = query["context"][field]
    return list(database[EXPERIENCES].find(match, {"embedding": 0}).sort("uses", -1).limit(1))


def _document_text(doc: dict[str, Any]) -> str:
    return f"{_tail(doc['error_text'])[-1500:]}\nfixed by: " + " ; ".join(doc["fix_commands"])


def _rerank(http: httpx.Client, query_text: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not candidates:
        return []
    response = http.post(
        "/rerank",
        json={"query": query_text[-4000:], "documents": [_document_text(doc) for doc in candidates],
              "model": RERANK_MODEL, "top_k": len(candidates)},
    )  # fmt: skip
    response.raise_for_status()
    return [{**candidates[row["index"]], "score": row["relevance_score"]} for row in response.json()["data"]]


def _judge(llm: AnthropicLLM, query: dict[str, Any], candidate: dict[str, Any]) -> bool:
    facts = _facts_from(query["context"])
    source = _facts_from(candidate["context"])
    prompt = (
        f"New machine: {facts.summary()}, step {query['context']['step_id']}. Its error:\n{_tail(query['error_text'])[-3000:]}\n\n"
        f"Candidate fix, verified on {source.summary()} for step {candidate['context']['step_id']} against this error:\n"
        f"{_tail(candidate['error_text'])[-1500:]}\nFix commands:\n" + "\n".join(candidate["fix_commands"])
    )
    reply = llm.complete(JUDGE_SYSTEM, prompt, JUDGE_MAX_TOKENS)
    return bool(json.loads(reply[reply.find("{") : reply.rfind("}") + 1])["relevant"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = replace(settings(), mongodb_db=args.db, llm_timeout_s=240.0)
    database = _client(args.db)
    embedder = VoyageEmbedder(config)
    llm = AnthropicLLM(config)
    http = httpx.Client(base_url=config.embedding_base_url,
                        headers={"Authorization": f"Bearer {config.voyage_api_key}"}, timeout=60)  # fmt: skip

    queries = _queries(database)
    rankings: list[dict[str, list[dict[str, Any]]]] = []
    for query in queries:
        text = _memory_text(query["error_text"], f"step {query['context']['step_id']} on {_facts_from(query['context']).summary()}")
        [vector] = embedder.embed([text], QUERY)
        filtered = _vector(database, vector, query["context"]["step_id"], query["hire"])
        unfiltered = _vector(database, vector, None, query["hire"])
        rankings.append(
            {
                "exact_same_machine": _exact(database, query, same_machine=True),
                "exact_any_machine": _exact(database, query, same_machine=False),
                "vector": filtered,
                "vector_unfiltered": unfiltered,
                "vector_rerank": _rerank(http, text, filtered),
                "vector_unfiltered_rerank": _rerank(http, text, unfiltered),
            }
        )

    labels = {(doc["query"], doc["candidate"]): doc["relevant"] for doc in database[LABELS].find()}
    pairs = {
        (str(query["_id"]), str(candidate["_id"])): (query, candidate)
        for query, ranking in zip(queries, rankings, strict=True)
        for ranked in ranking.values()
        for candidate in ranked[:POOL_DEPTH]
    }
    todo = [pair for pair in pairs if pair not in labels]
    print(f"{len(queries)} queries, {len(pairs)} pooled pairs, {len(todo)} to label", flush=True)

    def label(pair: tuple[str, str]) -> None:
        relevant = _judge(llm, *pairs[pair])
        database[LABELS].insert_one({"query": pair[0], "candidate": pair[1], "relevant": relevant})
        labels[pair] = relevant

    with ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as pool:
        list(pool.map(label, todo))

    answerable = [
        any(labels.get((str(query["_id"]), str(candidate["_id"]))) for ranked in ranking.values() for candidate in ranked[:POOL_DEPTH])
        for query, ranking in zip(queries, rankings, strict=True)
    ]

    def score(system: str, threshold: float | None) -> dict[str, float]:
        answered = correct = 0
        for query, ranking in zip(queries, rankings, strict=True):
            top = ranking[system][0] if ranking[system] else None
            if top is None or (threshold is not None and top.get("score", 1.0) < threshold):
                continue
            answered += 1
            correct += bool(labels.get((str(query["_id"]), str(top["_id"]))))
        total = sum(answerable)
        precision = correct / answered if answered else 0.0
        recall = correct / total if total else 0.0
        return {
            "answered": answered,
            "correct": correct,
            "precision_at_1": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
        }

    rerank_thresholds = [round(0.3 + 0.05 * step, 2) for step in range(13)]
    report = {
        "queries": len(queries),
        "answerable": sum(answerable),
        "labelled_pairs": len(pairs),
        "package_managers": sorted({query["context"]["package_manager"] for query in queries}),
        "systems": {
            "exact_same_machine": score("exact_same_machine", None),
            "exact_any_machine": score("exact_any_machine", None),
            "vector@0.8": score("vector", PRODUCTION_THRESHOLD),
            "vector_unfiltered@0.8": score("vector_unfiltered", PRODUCTION_THRESHOLD),
            "vector_top1_no_threshold": score("vector", None),
            "vector_rerank_top1": score("vector_rerank", None),
            "vector_unfiltered_rerank_top1": score("vector_unfiltered_rerank", None),
        },
        "unfiltered_threshold_sweep": {
            str(threshold): score("vector_unfiltered", threshold) for threshold in THRESHOLDS
        },
        "threshold_sweep": {str(threshold): score("vector", threshold) for threshold in THRESHOLDS},
        "rerank_threshold_sweep": {str(threshold): score("vector_rerank", threshold) for threshold in rerank_thresholds},
        "judge_model": config.llm_model,
        "judge_calls": llm.calls,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report["systems"], indent=2))


if __name__ == "__main__":
    main()
