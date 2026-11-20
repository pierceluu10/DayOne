# Benchmarks

Three measurements of the same engine that onboards hires in the demo, pointed at real open-source projects. Every number below is read back from Atlas by `bench/report.py` and `bench/retrieval.py`; the raw results are in `results/`.

## Setup

- **Repositories** (`repos.json`): 18 projects pinned to a commit, picked for setups their contributor guides leave incomplete. They span Python (click, Django REST framework, Pillow, tiktoken, safetensors, imbalanced-learn, redis-py, black), Ruby (rack, sinatra, jekyll, campfire on Rails 8), Go (gin, go-sqlite3, cobra), Node (express, node-canvas) and Rust (ripgrep). They include native extensions, Rust extensions built through maturin, cgo, Cairo and service dependencies.
- **Machines**: bare `ubuntu:22.04`, `debian:12` and `alpine:3.20` containers, each limited to 2 GB RAM and 3 CPUs. Nothing is preinstalled, not even git.
- **Workflow** (`steps.py`): base tools, clone at the pinned SHA, the project's documented setup commands, then its test suite. A hire counts as onboarded only when the tests exit 0 with no human help.
- **Guards against shortcuts**:
  - Tracked project files must be unmodified, so the agent cannot edit or delete tests.
  - Python packages must import from the checkout, so the agent cannot install a released copy from PyPI.
  - Each run gets 3,600 s and 5 reasoning attempts per step.
- **Model**: `claude-sonnet-5-5` ($2 / $10 per MTok). Embeddings are Voyage `voyage-3.5`; the rerank model is `rerank-2.5`.

## 1. Memory off vs on (`python -m bench.matrix`)

Each of the 54 repo × OS pairs is onboarded twice. Memory-off reads no memory and reasons every fix from scratch. Memory-on runs afterwards in the same database, so it can reuse the fixes saved by the memory-off pass and by its own earlier hires.

| | Memory off | Memory on |
|---|---|---|
| Onboarded (green tests, unassisted) | 49/54 (90.7%) | **51/54 (94.4%)** |
| LLM calls | 220 | **18 (−92%)** |
| LLM cost | $2.62 | **$0.36 (−86%)** |
| Fixes from memory / by reasoning | 0 / 125 | 126 / 0 |
| Median wall-clock | 137 s | 137 s |

- **Rescued by memory**: rack on Debian and Alpine. rack's suite refuses to run as root; the Ubuntu hire found the fix and the others reused it.
- **Failed with memory off and on**:
  - redis-py on Debian and Alpine: its suite needs a six-node Redis cluster.
  - campfire on Ubuntu: Ruby 3.4 plus Rails 8 system dependencies.
- **Wall-clock is unchanged** because it is dominated by compiling and downloading dependencies, which a remembered fix still has to do.

## 2. Learning curve (`--label curve --os-major --workers 1`)

24 hires in a row on a fresh database: 8 repos on Ubuntu, then the same 8 on Debian, then on Alpine.

| | First 8 hires | Last 8 hires | Change |
|---|---|---|---|
| Mean LLM cost per hire | $0.0413 | $0.0168 | **−59%** |
| Mean LLM calls per hire | 3.12 | 1.75 | −44% |
| Mean wall-clock per hire | 175 s | 63 s | −64% |

All 24 hires onboarded. The whole curve used 220k tokens. See `results/learning_curve.svg`.

## 3. Retrieval quality (`python -m bench.retrieval`)

- **Queries**: every distinct setup or test failure a benchmark hire hit, about 115 in all across both databases.
- **Candidates**: pooled from every retrieval system under test. Each (error, candidate fix) pair is labelled relevant or not by a Claude judge, and labels are cached so every system is scored on the same judgments.
- **No self-matches**: a hire's own fix is excluded from its own query.

Results on the matrix database (81 queries, 73 with a correct fix somewhere in memory, 510 labelled pairs):

| System | Precision@1 | Recall |
|---|---|---|
| Exact fingerprint, same machine | 100% | 23% |
| Exact fingerprint, any machine | 100% | 64% |
| **Vector search, filtered to the step, score ≥ 0.8 (production)** | **97.3%** | **97.3%** |
| Vector search, unfiltered, ≥ 0.8 | 88.8% | 97.3% |
| Unfiltered + Voyage rerank, top 1 | 88.9% | 98.6% |

- **On the learning-curve database** (34 queries, 205 pairs): vector search at 0.8 scores 100% precision and 97% recall. Exact matching on any machine reaches 67% recall.
- **Threshold**:
  - With the step filter, the 0.8 threshold never rejects a correct hit; precision and recall are flat from 0.5 to 0.875.
  - Without the step filter, the threshold matters: 0.85 gives 91.0% precision and 97.3% recall, 0.9 gives 95.8% and 94.5%, and 0.925 gives 98.2% and 75.3%.
- **Rerank**:
  - Reranking the filtered list changes nothing at the top.
  - A rerank score cut of 0.85 reaches 100% precision at 87.7% recall.
  - So the step filter does the work a reranker would.

## Reproduce

```bash
uv run python -m bench.matrix --label matrix-off --memory off --db dayone_bench --workers 3
uv run python -m bench.matrix --label matrix-on  --memory on  --db dayone_bench --workers 3
uv run python -m bench.matrix --label curve --memory on --db dayone_curve --workers 1 --os-major \
  --repos click,django-rest-framework,pillow,black,rack,sinatra,gin,express
uv run python -m bench.retrieval --db dayone_bench --out bench/results/retrieval.json
uv run python -m bench.report
```

Re-running skips any pair already recorded under the label; per-run logs go to `.dayone/bench/<label>/`.
