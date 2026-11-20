"""Summarise recorded runs from Atlas into bench/results: the memory off/on matrix and the learning curve.

    python -m bench.report
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from statistics import mean, median
from typing import Any

from pymongo import MongoClient

from dayone.config import settings

from bench.run_one import RUNS

RESULTS = Path(__file__).with_name("results")
MATRIX_DB, CURVE_DB = "dayone_bench", "dayone_curve"
OFF, ON, CURVE = "matrix-off", "matrix-on", "curve"
WINDOW = 8  # hires per window when comparing the start of the curve with its end
CHART_W, CHART_H, PAD = 720, 320, 48


def _runs(db: str, label: str) -> list[dict[str, Any]]:
    config = replace(settings(), mongodb_db=db)
    with MongoClient(config.mongodb_uri, tz_aware=True) as client:
        return list(client[db][RUNS].find({"label": label}, {"_id": 0}).sort("sequence", 1))


def _rate(runs: list[dict[str, Any]]) -> float:
    return round(sum(run["ok"] for run in runs) / len(runs), 4) if runs else 0.0


def _totals(runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "runs": len(runs),
        "onboarded": sum(run["ok"] for run in runs),
        "success_rate": _rate(runs),
        "llm_calls": sum(run.get("llm_calls", 0) for run in runs),
        "cost_usd": round(sum(run.get("cost_usd", 0.0) for run in runs), 2),
        "median_elapsed_s": median(run["elapsed_s"] for run in runs) if runs else 0,
        "fixed_from_memory": sum(run.get("fixed_from_memory", 0) for run in runs),
        "fixed_by_reasoning": sum(run.get("fixed_by_reasoning", 0) for run in runs),
    }


def _by(runs: list[dict[str, Any]], key: str) -> dict[str, float]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        groups[run.get(key, "?")].append(run)
    return {group: _rate(members) for group, members in sorted(groups.items())}


def matrix() -> dict[str, Any]:
    off, on = _runs(MATRIX_DB, OFF), _runs(MATRIX_DB, ON)
    pairs = {(run["repo"], run["machine"]): run for run in off}
    rescued = [f"{run['repo']}/{run['machine']}" for run in on if run["ok"] and not pairs.get((run["repo"], run["machine"]), {}).get("ok")]
    return {
        "repos": len({run["repo"] for run in off}),
        "machines": sorted({run["machine"] for run in off}),
        "memory_off": {**_totals(off), "by_stack": _by(off, "stack"), "by_machine": _by(off, "machine")},
        "memory_on": {**_totals(on), "by_stack": _by(on, "stack"), "by_machine": _by(on, "machine")},
        "rescued_by_memory": rescued,
        "failed_both": [
            f"{run['repo']}/{run['machine']}" for run in on if not run["ok"] and not pairs.get((run["repo"], run["machine"]), {}).get("ok")
        ],
        "runs": [{key: run.get(key) for key in ("label", "repo", "machine", "ok", "llm_calls", "cost_usd", "elapsed_s", "fixed_from_memory", "fixed_by_reasoning")} for run in off + on],
    }


def curve() -> dict[str, Any]:
    runs = _runs(CURVE_DB, CURVE)
    first, last = runs[:WINDOW], runs[-WINDOW:]
    first_cost, last_cost = mean(run.get("cost_usd", 0.0) for run in first), mean(run.get("cost_usd", 0.0) for run in last)
    first_calls, last_calls = mean(run.get("llm_calls", 0) for run in first), mean(run.get("llm_calls", 0) for run in last)
    return {
        "hires": len(runs),
        "success_rate": _rate(runs),
        "first_window_mean_cost_usd": round(first_cost, 4),
        "last_window_mean_cost_usd": round(last_cost, 4),
        "cost_reduction": round(1 - last_cost / first_cost, 4) if first_cost else 0.0,
        "first_window_mean_llm_calls": round(first_calls, 2),
        "last_window_mean_llm_calls": round(last_calls, 2),
        "llm_call_reduction": round(1 - last_calls / first_calls, 4) if first_calls else 0.0,
        "first_window_mean_elapsed_s": round(mean(run["elapsed_s"] for run in first), 1),
        "last_window_mean_elapsed_s": round(mean(run["elapsed_s"] for run in last), 1),
        "total_tokens": sum(run.get("input_tokens", 0) + run.get("output_tokens", 0) for run in runs),
        "per_hire": [
            {key: run.get(key) for key in ("sequence", "repo", "machine", "ok", "llm_calls", "input_tokens", "output_tokens", "cost_usd", "fixed_from_memory", "fixed_by_reasoning", "elapsed_s")}
            for run in runs
        ],
    }


def chart(per_hire: list[dict[str, Any]]) -> str:
    """Cost per hire and its running mean, as a self-contained SVG."""
    costs = [hire["cost_usd"] for hire in per_hire]
    top = max(costs) or 1.0
    width, height = CHART_W - 2 * PAD, CHART_H - 2 * PAD
    step = width / max(len(costs) - 1, 1)

    def point(index: int, value: float) -> str:
        return f"{PAD + index * step:.1f},{PAD + height - value / top * height:.1f}"

    running = [mean(costs[: index + 1]) for index in range(len(costs))]
    bars = "".join(
        f'<rect x="{PAD + index * step - 6:.1f}" y="{PAD + height - cost / top * height:.1f}" width="12" '
        f'height="{cost / top * height:.1f}" fill="#9fb7d9"><title>hire {index + 1}: ${cost:.3f}</title></rect>'
        for index, cost in enumerate(costs)
    )
    line = " ".join(point(index, value) for index, value in enumerate(running))
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {CHART_W} {CHART_H}" font-family="sans-serif" font-size="12">'
        f'<rect width="100%" height="100%" fill="white"/>'
        f'<text x="{PAD}" y="24" font-size="14">LLM cost per onboarding as memory fills (bars), running mean (line)</text>'
        f'{bars}<polyline points="{line}" fill="none" stroke="#1f4e8c" stroke-width="2"/>'
        f'<line x1="{PAD}" y1="{PAD + height}" x2="{PAD + width}" y2="{PAD + height}" stroke="#333"/>'
        f'<text x="{PAD}" y="{CHART_H - 12}">hire 1</text><text x="{PAD + width - 40}" y="{CHART_H - 12}">hire {len(costs)}</text>'
        f'<text x="4" y="{PAD + 4}">${top:.2f}</text><text x="4" y="{PAD + height}">$0</text></svg>'
    )


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    summary = {"matrix": matrix(), "curve": curve()}
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (RESULTS / "learning_curve.svg").write_text(chart(summary["curve"]["per_hire"]))
    printable = {
        "matrix": {key: value for key, value in summary["matrix"].items() if key != "runs"},
        "curve": {key: value for key, value in summary["curve"].items() if key != "per_hire"},
    }
    print(json.dumps(printable, indent=2, default=str))


if __name__ == "__main__":
    main()
