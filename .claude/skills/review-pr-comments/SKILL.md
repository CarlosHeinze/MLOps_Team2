---
name: review-pr-comments
description: Triage the review comments on a pull request of this repo (Copilot or teammates). Judge each one for an academic MLOps project that is not meant for production, reply with a note when it does not apply, apply the quick fixes, and plan the larger ones. Use when asked to check, address or answer the review of a PR.
argument-hint: "<PR number>"
---

# Triage the review of a PR

This repo is a course project (MLOps, Phase 1 data pipeline, Phase 2 models). It is graded on
correct ML practice, reproducibility, tracking and clean code. It is **not** deployed, so a
comment is judged on whether it matters *for this project*, not for a production service.

## 1. Read everything

```
.venv/Scripts/python.exe .claude/skills/review-pr-comments/github.py fetch <PR>
```

This prints the review summaries, the inline comments (with their ids) and the conversation
comments. Copilot's summary can hold findings that have **no inline comment** (the "Review
findings" column of its table, often lower-severity votes). Triage those too.

Check out the PR branch (`git status`, `git log`) and read the code each comment points at
before judging it. Also read the issue the PR closes, its acceptance criteria and
`references/ml-canvas.md`, where the project's decisions are recorded.

## 2. Judge each comment

**Valid: it breaks something the project is graded on**
- wrong results: data leakage (a random split, a fit that sees test hours), wrong metrics, a
  crash on the real data
- reproducibility and lineage: MLflow runs, digests, seeds, the pipeline commands
- an acceptance criterion of the issue that is not met
- tests that pass for the wrong reason, or a real behaviour with no test
- a project standard that is broken: logic in notebooks (`def`s), comments that only repeat the
  code, decisions missing from the canvas, `make lint` / `make test` failing

**Not valid here: note it and move on**
- production hardening: authentication, retries, rate limits, concurrency, horizontal scaling,
  deployment and monitoring infrastructure beyond what the canvas plans
- speed-ups that save seconds on a step run by hand (measure before arguing either way)
- defensive checks for inputs the code never receives, or ones already rejected upstream
- style preferences that contradict the existing code, or more comments and docstrings than
  the project's comment policy allows
- anything already settled in the canvas or the issue: cite where

A comment can be valid in principle but not worth the change. Say so, and say what would make
it matter (for example "if the backtest ever uses a test month before 2025-10").

## 3. Act on it

| Verdict | Action |
|---|---|
| Not valid | Draft a reply: why it does not apply here, with facts (file:line, a measurement, the canvas section). No code change. |
| Valid, quick (a few lines plus a test) | Make the change in the style of the surrounding code. Add or adjust a test. Record any decision in the canvas, not in a comment. Run `make lint` and `make test` (`PYTHON_INTERPRETER=.venv/Scripts/python.exe`). Rerun a notebook only if its outputs change. Draft a reply that says what changed. |
| Valid, large (new design, several modules, reruns of the heavy notebooks) | Do not start it. Write a plan: the problem, the files, the tests, and whether it belongs in this PR or a later issue (the open issues on GitHub say which one owns what). Draft a reply that points to the plan or proposes the follow-up issue. |

Keep the replies short and in English. Write them as the PR author, so no "as an AI". One reply
per inline comment. Put findings without an inline comment in one top-level PR comment.

## 4. Report, then publish on approval

Show the user a table (comment, verdict, action) with the diff, the test results and the
draft replies. Committing, pushing and posting are outward-facing, so do them only when the
user says so:

1. Commit the fixes on the PR branch (conventional message, `fix: ...`) and push.
2. Put the replies in a JSON file in the scratchpad, `{"<inline comment id>": "text", "pr":
   "top-level text"}`. Mention the fix commit's short SHA in each reply.
3. Run `.venv/Scripts/python.exe .claude/skills/review-pr-comments/github.py reply <PR> <file>`.
