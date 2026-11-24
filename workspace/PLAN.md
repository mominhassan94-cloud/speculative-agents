# Project plan: Speculative Tool Execution

For Momin and Khizar. Written 2026-11-15, final deliverable due 2026-12-01 (16 days).
Everything marked *(inferred)* is my reading of the repo and the research report, not a verified fact.

---

## Status, Nov 24: now a solo project (Khizar owns every track)

| Track | Status | Where |
|---|---|---|
| Merge Gemini branch | done, PR #2 | `khizar/gemini-migration` |
| B. Tasks: 26 with train/held-out split | done | `khizar/tasks-heldout` |
| A. Sweeps (H1, H2, break-even) | done: `sweep.py`, `plots.py` | `khizar/sweeps` |
| F. Stale cache (H8) | done: hazard shown under the old cache, fixed by per-step cache or scope invalidation | `khizar/stale-cache` |
| G. Adversary, canary, lossless (H7, H9, H10) | done: `adversary.py` | `khizar/adversary` |
| C. Mined speculator (H3, H4) | built: `mined.py`, offline held-out comparison | `khizar/adversary` (next PR) |
| D. Live Gemini (H5, H6) | setup 3 train run in progress (Pro agent, Flash-Lite guesser, billing on) | |
| E. Threshold sweep (H6) | next | |
| Charts, README results, write-up | Nov 29-30 | |

**Correction to section 1.3:** a hit saves `min(D, T - s)`, not `min(T, D - s)`. The guess starts `s` after the
result and finishes at `s + D`; the agent asks at `T`, so the call is ready at `max(T, s + D)` instead of `T + D`.
With one backend slot, a miss costs `max(0, s + D - T)`. `sweep.py` tests this form.

**Correction to gap 5:** the stale-hit hazard was real on `main` (unused guesses lived for the whole run). The
merged runtime drops unused guesses at every real call, which closes it; `hazard.py` shows both.

---

## 0. Where we actually are (as of Nov 15)

**Built and working (on `add-workspace-folder`):**
- Mock tools with latency, read-only / side-effect flags, reversible "prepare" variants (`mock_tools.py`)
- Scripted agent brain + real-LLM brain (`agent.py`)
- Rule and Gemini (LLM) speculators (`speculator.py`); Gemini agent brain (`agent.py`, `llm_client.py`). Gemini is the only LLM provider.
- Runtime: cache, safety gate, confidence threshold, provenance check, prepare/rollback, event log, wasted tool-seconds (`runtime.py`)
- Eval harness: 20 reps default, shuffled mode order, paired speedup with bootstrap CI (`run_eval.py`)
- Research report with 10 falsifiable hypotheses, H1 to H10 (`workspace/research_report.md`)

**Gaps that matter (these are the real work):**
1. **Only 6 tasks exist.** The README promised ~50 by Nov 1. The research report says task *count* matters less than a **held-out split**, so we should aim for about 25-30 good tasks, not 50 shallow ones.
2. **`RuleSpeculator` is written against the same plans it is scored on**, so it ties the oracle. Its results say nothing about generalization (report, section 3).
3. **No speculator that learns from data.** A mined (pattern-based) speculator is the missing middle rung.
4. **No sweeps.** Savings are capped by think time (`--think`) and tool latency (`--scale`). Any single percentage without a sweep is an artifact of one constant.
5. **Stale-hit hazard is still open** *(inferred from `runtime.py`)*: the cache has no TTL and is not invalidated when a write tool runs, so a read cached before `book_flight` can be served after it.
6. **No live-LLM results.** All committed runs are offline with the scripted brain.
7. ~~`khizar/gemini-migration` diverged on Nov 1~~ Merged with `main` on Nov 23 (Gemini is the only provider).
8. README milestones are stale (Nov 15 "end to end" is today).

---

## 1. What we need to understand first

Do these in order. Each has a "you understand it when..." test. Budget: about 6-8 hours each, spread over the first 3 days, mostly overlapping with Phase 0 and 1 work.

### 1.1 The core mechanism (both, day 1)
- Read the repo top to bottom in this order: `mock_tools.py`, `agent.py`, `runtime.py`, `speculator.py`, `tasks.py`, `run_eval.py`. About 940 lines total.
- Run `python demo.py` and `python run_eval.py --reps 3`. Read the printed summary line by line.
- **You understand it when** you can explain, without looking, why a hit saves `min(tool_time, think_time)` and not the whole tool time.

### 1.2 The three papers (split, then teach each other)
| Paper | Who reads closely | Take away |
|---|---|---|
| Speculative Actions (arXiv 2510.04371) | Khizar | The speedup bound: even a perfect speculator cuts at most about half the runtime |
| PASTE (arXiv 2603.18897, cite **v3**) | Momin | Pattern mining from traces, argument mappers, why blind speculation can hurt |
| LLMCompiler (arXiv 2312.04511) | skim, both | The non-speculative baseline: parallelism beats guessing when calls are independent |

Each person writes a **half-page summary in their own words** into `workspace/notes/` and presents it in 15 minutes at the first sync. Teaching it is the test.

### 1.3 The latency model (both, derive by hand)
Per step with tool time `D`, think time `T`, speculator latency `s`: saving on a hit is about `min(T, D - s)`. Whole-run saving is about `p x min(T, D - s)` where `p` is hit rate.
- **You understand it when** you can predict, before running, what happens to the speedup if you double `--think`, halve `--scale`, or drop hit rate to 25%. Then check against the sweep (this is H1, H2).

### 1.4 Safety concepts (Momin leads, Khizar reviews)
- Read-only is not the same as safe to run early (staleness, billing, audit logs, cursors).
- Three levels of "lossless": same answer, same transcript, same world state.
- Why a speculator that reads attacker-controlled text is a security issue (injection canary, provenance rule).
- **You understand it when** you can describe one concrete way the current `runtime.py` serves a wrong cached result.

### 1.5 Statistics basics (Khizar leads)
Paired comparison, geometric-mean speedup, bootstrap confidence intervals, why one run proves nothing. `run_eval.py` already does this; read `summarize()` and `bootstrap_ci()` and be able to explain each number in the output.

### 1.6 Things we should NOT trust yet
The report itself lists these (section 5). Before any number goes in a slide: the ICLR acceptance claim is unconfirmed, the PASTE title in the README is outdated (v3 differs), several 2026 papers were read through search snippets only, and the latency table is the report author's estimates. Re-check against the papers.

---

## 2. What we're really trying to learn (the takeaways)

The honest framing from the report: "speculation saves time" is already published. Our contribution is **a map of when it pays and when it loses, plus a safety result that survives an adversary.** By Dec 1 we should be able to say, with data:

1. **Where the break-even is:** the region of (hit rate, tool latency, think time, speculator latency) where speculation wins vs. loses. One heatmap.
2. **How much of the oracle's savings each speculator captures** (rules vs mined vs LLM), on tasks the speculator was *not* written against.
3. **The cost side:** wasted calls and wasted tool-seconds per second saved. The chart the README promised: time saved vs. calls wasted.
4. **Whether the safety gate holds** under an adversarial speculator and a stale-cache attack, and an honest account of where it did not.
5. **A negative result we are willing to report.** If the LLM speculator is not worth its cost on fast tools, that is a finding, not a failure.

Personal takeaways worth deliberately collecting (keep a running `workspace/notes/learnings.md`):
- How to design an experiment that can falsify you (H1 to H10).
- Concurrency in practice: futures, caches, races, cleanup of unused work.
- Reading research critically: what a number actually counts.
- Cost/latency trade-offs of LLM systems, with real numbers.

---

## 3. The plan, phase by phase

Milestone dates are the README's, adjusted. **Proposed edit to the README status list:** replace "~50 tasks" with "~25-30 tasks with a held-out split".

### Phase 0: Align and unblock (Nov 15-16)
Together, one 90-minute session.
- [ ] Walk through this plan; change anything either of you disagrees with.
- [ ] **Merge `khizar/gemini-migration` into the working branch now.** Expect conflicts in `runtime.py` and `speculator.py`; resolve together, since both of you need to know the merged result. Then rerun `python run_eval.py --reps 3` to confirm nothing changed in offline results.
- [ ] Open a PR for the merged branch into `main`; from here on, all work goes through short-lived feature branches (section 5).
- [ ] Agree on the one-sentence claim from the report's first paragraph. Everything we do should serve it.
- [ ] Update README status/milestones to match reality.

### Phase 1: Make the evidence real (Nov 17-20)
| Track | Owner | Work | Tests |
|---|---|---|---|
| A. Sweeps | Momin | `CorruptedOracle(p)` wrapper, simulated speculator delay `s`, `--think` / `--scale` sweep driver, lognormal latencies | H1, H2 |
| B. Tasks | Khizar | Grow tasks from 6 to about 25-30; mark each as `train` or `heldout`; include vague tasks and tasks with write steps. Write held-out tasks with **different tool orderings** so rules cannot just match | H3 |

Done when: a single plot of measured saving vs. predicted `p x min(T, D - s)`, and a task file with a clear train/held-out split. Neither of you looks at the held-out tasks while building speculators.

### Phase 2: Speculators that can fail (Nov 21-25)
| Track | Owner | Work | Tests |
|---|---|---|---|
| C. Mined speculator | Momin | `NGramSpeculator`: key on (last tool, status), learn next-tool counts and argument templates (e.g. "url = first result url") from baseline-run logs on **train tasks only** | H3, H4 |
| D. Live LLM | Khizar | Run the Gemini speculator (`LLMSpeculator`) on held-out tasks, 10+ reps, log name-accuracy vs exact-match accuracy, add a "copy args only from context" check, calibrate confidence | H5, H6 |
| E. Threshold sweep | together, Nov 24 | Sweep confidence 0 to 1 for all speculators; build the precision / waste / speedup plot | H6 |

Done when: a table of hit rate, precision, waste and speedup per speculator, on **held-out tasks only**.

### Phase 3: Safety (Nov 26-28)
Light days around Thanksgiving if either of you is celebrating; Track F can run alone.
| Track | Owner | Work | Tests |
|---|---|---|---|
| F. Stale cache | Momin | Hazard task (read, write same resource, read again); confirm stale hit happens today, then add scope-versioned invalidation or TTL and show it fixed | H8 |
| G. Adversary | Khizar | `AdversarialSpeculator` guessing every write tool at confidence 1.0; injection canary in a search result; transcript hash + world-state snapshot comparison across modes | H7, H9, H10 |

Done when: every H7-H10 result is reported as pass or fail, including failures.

### Phase 4: Freeze and tell the story (Nov 29 - Dec 1)
- **Nov 29: code freeze.** Rerun all committed experiments from scratch on the final code with fixed seeds; commit CSVs.
- Nov 29-30: charts (time saved vs wasted calls; oracle-gap per speculator; break-even heatmap), README results section, write-up, demo script. Split by section, then cross-edit each other's.
- Nov 30: dry-run the demo end to end, once, together.
- Dec 1: submit.

### If we fall behind (cut order)
1. Drop self-speculation / streaming early dispatch (already a stretch goal).
2. Drop tau-bench replay (`replay_eval.py`) and say so in the write-up.
3. Drop live Gemini-agent (setup 3) runs to a small demo subset; keep the Gemini speculator runs.
4. **Never cut:** held-out split, sweeps, the stale-cache and adversary tests. Those are the contribution.

---

## 4. Day-to-day: separate and together

### Ownership
- **Momin:** sweeps, mined speculator, stale-cache fix (runtime / eval-infrastructure side).
- **Khizar:** task set, live LLM runs and calibration, adversary and canary tests (task and LLM side).
- **Shared:** merge, threshold sweep, final charts, write-up, demo.
- Rule: **one owner per file at a time.** `runtime.py` and `run_eval.py` are the hot spots; announce in chat before touching a file you don't own.

### Solo days (default)
1. **Start (10 min):** pull `main`, read the last day's chat, look at the open PRs.
2. **Work block (2-3 hours, one track only):** write the test or experiment first, then the code.
3. **Log (10 min):** append to `workspace/notes/learnings.md`: what you did, what surprised you, what you still don't understand.
4. **End:** push your branch; post a 3-line async update (done / next / blocked).

### Sync cadence
- **Daily async update** in chat, 3 lines. No meeting.
- **Twice-weekly 30-minute call** (suggested: Tue and Fri): demo what ran, compare results, decide next 3 days.
- **Pairing sessions (about 1-2 hours each), only for these:** the merge (Nov 15-16), the threshold sweep (Nov 24), anything touching `runtime.py` cache semantics, and the final freeze (Nov 29).
- **Review swap:** every PR is reviewed by the other person within a day. Reviewing is also how you learn the half you didn't write.

### Git workflow
- `main` = always runs offline (`python run_eval.py --reps 3` passes).
- Branch per track: `momin/sweeps`, `khizar/tasks-heldout`, etc. Small PRs (under about 300 lines).
- Commit result CSVs, but never API keys. Keys stay in environment variables.
- Never rewrite shared history; merge or rebase only your own branch.

### Keeping experiments honest
- Fixed seeds (`--seed`), recorded in the CSV filename or a sidecar note.
- Every result we plot should be reproducible with one command written in the PR description.
- Held-out tasks stay unseen until the first held-out evaluation run, and we do not tune on them afterward.

---

## 5. Definition of done (Dec 1)

- [ ] 25-30 tasks, train/held-out split, committed
- [ ] H1-H10 each have a recorded result (supported, refuted, or not tested, and why)
- [ ] Charts: time saved vs wasted calls; oracle-gap per speculator; break-even heatmap
- [ ] Stale-cache hazard demonstrated and fixed (or honestly reported as unfixed)
- [ ] README: updated status, results section, corrected citations (PASTE v3, no unconfirmed ICLR claim)
- [ ] One-command reproduction of every figure
- [ ] Both of us can explain every number in the write-up unaided

---

## 6. Risks

| Risk | Mitigation |
|---|---|
| Gemini merge conflict eats a day | Do it first, together, Nov 15-16 |
| Live LLM runs cost money or hit rate limits | Cap reps for live cells, log token spend, run on a small task subset first |
| Results are boring (speculation barely helps) | Report the map of where it loses; that is still the contribution |
| Rules look great on held-out tasks | Then the held-out tasks are too similar; rewrite them with different orderings before drawing any conclusion |
| One of us is unavailable for a few days | Daily updates and one-owner-per-file mean the other can continue; keep `learnings.md` current |
| Numbers from papers are misquoted | Verify against the arXiv text before they go in a slide |

---

## 7. Other ways to approach this (summary)

The plan above is shape **A, "the map"**. Full detail lives in [`DIRECTIONS.md`](DIRECTIONS.md); this is the short version.

**Ways to cut agent time, beyond speculation:** parallel independent calls, plan-then-execute, macro tools, batched calls, model routing, streaming early dispatch, prompt caching, result caching, hedged requests. Speculation only wins on *dependent chains* where the next argument depends on a result the agent hasn't reasoned over yet.

**Extra hypotheses (H11 to H24):** parallel vs speculation crossover, p95 vs mean gain, decay with chain depth, cost-aware gating, top-k speculation, online learning, first-call prediction, macro tools vs speculation, stale-hit growth with TTL, reversible-write speculation, and more.

**Other project shapes:**

| Shape | Question |
|---|---|
| A. The map (current) | When does speculation pay and when does it lose? |
| B. Head-to-head | Speculation vs parallel vs macro tools vs plan-execute, same harness |
| C. Cost-aware policy | Expected-value gate vs fixed threshold |
| D. Attack surface | What can a poisoned observation make a speculator do? |
| E. Speculative writes | Hold-then-commit for side-effecting tools |
| F. Replay study | How predictable are next calls in real trajectories? |
| G. Real tools | Do the mock conclusions hold with real APIs? |

**Suggested default:** A + D. Decide for real after the sweeps (Nov 20), when real numbers show which direction is richest.
