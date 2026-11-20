# Day One

## What it solves

Agents solve many problems on engineering teams, just not the beginning. Every new hire loses their first days to the same setup: outdated onboarding docs, missing system packages, the wrong runtime, a settings file nobody mentioned. Each one hits the problems the last hire already solved, solves them alone, and the fix disappears into a Slack thread. Day One makes onboarding a solved problem: the agent does it, and every fix one hire's agent learns is there for the next.

## How it works

Day One is an agent harness that gets better with every hire. It follows your team's onboarding workflow end to end: it signs into your tools in a real browser through Playwright, sets up the machine in a real terminal, and verifies each step actually took effect. Every error it hits and the fix that worked is stored in MongoDB Atlas, so the next hire's agent finds it with an exact match or Atlas Vector Search and applies it instantly, even on an operating system no one has used before.

A new hire runs one command in their terminal:

```bash
dayone start --hire <name>
```

and is onboarded autonomously: accounts, SSH keys, repository, runtime, system packages, settings, database and a green test suite. If the agent crashes or is killed, `dayone resume` picks up exactly where it stopped from the ledger in Atlas, with nothing done twice. If the repository or a web UI changes mid-run, it detects the drift, re-verifies what is affected and adapts.

`demo_repo/` (`acme-api`) is a sample service with the kind of undocumented setup snags every team has; its README mentions none of the four the agent has to find.

## Run it

Needs Docker, [uv](https://docs.astral.sh/uv/), and a MongoDB Atlas cluster with Vector Search.

```bash
cp .env.example .env        # fill in MONGODB_URI, ANTHROPIC_API_KEY, VOYAGE_API_KEY and the local Gitea passwords
uv sync
uv run playwright install chromium
scripts/up.sh               # local Gitea 1.24.7 seeded with org acme, repo acme/acme-api, five hires and their Welcome issues
uv run dayone serve         # the page on http://localhost:8000: name, machine, Start
```

Or from the command line:

```bash
uv run dayone start --hire ada --machine ubuntu     # ubuntu | debian | alpine
uv run dayone resume --hire ada                     # after a crash or kill -9
uv run dayone reset ada                             # container, ledger and Gitea state for one hire
scripts/push_change.sh                              # commit a new dependency to acme-api while a hire is mid-run
scripts/reset.sh                                    # put Gitea and acme-api back to the seed
```

## Experience memory (Atlas Vector Search)

When a step's check still fails after `run()`, the agent:

1. Captures the error (the last 4,000 characters, where pip and apt put the cause) and its context: step, OS family, package manager and tool versions.
2. Fingerprints the error Sentry-style, stripping paths, versions and timestamps.
3. Looks for an exact match: same fingerprint, same package manager. On a match it replays that fix with no LLM call.
4. On a miss, runs `$vectorSearch` over `experiences` filtered to `outcome: "verified"`, with a cosine score of at least 0.8. A hit from another machine is adapted by the LLM; the page shows the similarity score and the hire it came from.
5. Otherwise, or if the retrieved fix fails, asks Claude to reason, with the nearest experiences as hints. It gets at most 3 attempts, and each attempt is fed the previous commands' output.
6. Saves the fix only once the step's check passes. The record holds the fingerprint, error text, context, fix commands, `verified_by_check` (the evidence string) and the hires that used it.

When the same fix has helped two or more hires, the agent opens an issue on `acme-api` proposing the README change.

Collections: `hires` (lease, container, run), `steps` (unique on hire and step), `events` (ordered by timestamp and sequence; the page streams these over SSE) and `experiences` (vector index on `embedding`, created in code).

## Benchmarks (18 open-source repos × 3 operating systems)

The same engine, given only each project's own documented setup, onboarded 18 real open-source repositories across Python, Ruby, Go, Node and Rust on bare Ubuntu, Debian and Alpine containers. A hire counts only if the project's test suite goes green with no human help. Method and full results are in [bench/README.md](bench/README.md).

| | Memory off | Memory on |
|---|---|---|
| Onboarded (54 repo × OS pairs) | 90.7% | **94.4%** |
| LLM calls | 220 | **18 (−92%)** |
| LLM cost | $2.62 | **$0.36 (−86%)** |

- **Learning curve**: over 24 hires in a row, mean LLM cost per hire fell 59% between the first 8 and the last 8, and wall-clock time fell 64%. All 24 were onboarded.
- **Retrieval quality**: across about 115 real failures and 715 judged (error, fix) pairs, step-filtered Atlas Vector Search at the 0.8 threshold returned a correct fix first 97–100% of the time, with 97% recall. Exact fingerprint matching reached 64–67% recall.

## Benchmarks (4 sequential hires)

Four hires run one after another on a freshly reset Gitea, `acme-api` at c9d8d7f, between 15:08 and 15:29. Measured with `uv run python tests/acceptance/measure.py`, which reads only Atlas.

| Hire | Machine | LLM calls | Fixed by reasoning | Fixed from memory | Errors | Elapsed |
|---|---|---|---|---|---|---|
| ada | ubuntu | 7 | 4 | 0 | 4 | 2m 55s |
| grace | debian | 4 | 0 | 4 (similar error, adapted from ubuntu) | 4 | 3m 54s |
| linus | ubuntu | 0 | 0 | 4 (exact match) | 4 | 2m 36s |
| ken | alpine, never seen | 3 | 0 | 3 (similar error, adapted from ubuntu and debian) | 3 | 8m 40s |

The headline is the LLM column. Ada solved all four snags by reasoning, which took 7 calls because three steps needed a second attempt. Grace on Debian found each of ada's fixes by vector search (similarity 0.92 to 0.95) and spent one call per fix adapting it. Linus matched every error exactly and made no LLM call at all. Ken on Alpine, a machine no hire had used, got ada's apt fix for psycopg2 by vector search (similarity 0.94), adapted to `apk add --no-cache postgresql-dev build-base python3-dev musl-dev`. The settings and database fixes came from grace's Debian run the same way. Alpine ships Python 3.12, so the runtime snag never fired. Every hire still ended at "13 passed of 13, exit 0 at c9d8d7f".

Wall-clock time barely moves, and that is honest, not a bug: the time is spent installing Python 3.12 and compiling psycopg2, which a remembered fix still has to do. In linus's run those two steps took 31 s and 42 s of 156 s; reasoning saved about 20 s on the same OS. Ken took 8m 40s because psycopg2 has no musl wheel and compiled for 409 s after memory had already supplied the right packages. Grace was slower than ada because Debian's installs ran longer (86 s for the system package step) and each adapted fix cost an LLM call. Apt mirror speed alone moves a run by 10 to 15 s (`t_base_tools`: 15 s for ada, 28 s for linus).

Once linus applied the same four fixes as ada, the agent opened README issues #10 to #13 on `acme-api`, one per snag. Example: "README: t_settings on ubuntu needs `cp /root/acme-api/settings.example.toml /root/acme-api/settings.toml`".

From the acceptance database (`dayone_acceptance`) runs today:

- Hire 2 against hire 1 (test 4, both ubuntu): 200.8 s, 4 errors, 0 from memory, 6 LLM calls, against 135.6 s, 4 errors, 4 from exact memory, 0 LLM calls.
- Alpine, never seen before (test 5, 15:20 run): three of barbara's ubuntu fixes were retrieved by vector search (similarity 0.94, 0.94, 0.93). The `apt-get install` for psycopg2 became `apk add --no-cache postgresql-dev python3-dev build-base musl-dev`; the settings copy and db init carried over unchanged.
- `push_change.sh` mid-run (test 6): drift was detected (2955521 → 2143c85) and 8 steps were marked stale and re-verified. The new python-ldap error was fixed by reasoning with `apt-get install -y libldap2-dev libsasl2-dev` after the remembered fix failed. The run ended at 14 passed of 14.
