# Directions, hypotheses, and other ways to cut agent time

Companion to `PLAN.md`. The plan commits to one path. This file lays out the alternatives so we can choose on purpose, borrow ideas, or pivot if a track dies.

Honesty note: effort and "expected result" columns are my estimates *(inferred)*, not measurements. Paper numbers come from `research_report.md`, which flags its own weak sources.

---

## 1. Every way to reduce agent wall-clock time

Speculation is one of about ten levers. Knowing the others tells us what our result should be compared against, and where speculation is the wrong tool.

| # | Lever | Idea | Cuts which wait | Needs guessing? | Risk |
|---|---|---|---|---|---|
| 1 | **Speculative tool execution** (ours) | Guess next call, start early, keep on exact match | Tool latency hidden behind think time | Yes | Wasted calls, stale hits |
| 2 | **Parallel independent calls** (LLMCompiler) | If calls don't depend on each other, run together | Tool latency | No | Needs a plan up front |
| 3 | **Plan then execute** (ReWOO) | One LLM call plans all steps with placeholders; tools run without the LLM in the loop | LLM steps | No | Brittle when plans need results |
| 4 | **Macro tools** | Add a composite tool such as `search_and_open_top(k)` so one step does two | LLM steps | No | Tool design effort, less flexible |
| 5 | **Batched tool calls per turn** | Let the model emit several calls in one step | LLM steps | No | Only helps when model knows the batch |
| 6 | **Model routing / cascade** | Small fast model for easy steps, big model only when needed | LLM step time | No | Quality loss on routing errors |
| 7 | **Streaming early dispatch** | Launch a tool as soon as its arguments finish streaming, before the LLM response ends | Generation tail | No (uses the actor's own args) | Only saves the tail |
| 8 | **Prompt caching / shorter context** | Reuse the prompt prefix; trim long observations | LLM step time | No | Provider-dependent |
| 9 | **Result caching across runs** | Remember tool results for repeated queries | Tool latency | No | Staleness |
| 10 | **Hedged requests** | Send a duplicate request after a delay and take the first reply | Tail latency (p95) | No | Doubles some load |
| 11 | **Speculative decoding** (server side) | Draft tokens with a small model, verify with the big one | LLM token time | No | Out of our reach: it lives in the serving stack |

**What this tells us:**
- Levers 2, 3, 4 and 5 beat speculation when the next step is already known. Speculation earns its place only on **dependent chains where the next argument depends on a result the agent hasn't reasoned over yet** (search, then open a result).
- Lever 7 is speculation with zero prediction error but a small prize.
- Any honest write-up should include at least lever 2 as a baseline.

---

## 2. More hypotheses (extends H1 to H10 in the research report)

Numbering continues from H10. "How we falsify" is written so the harness can actually test it.

| ID | Hypothesis | How we falsify it | Expected *(inferred)* |
|---|---|---|---|
| H11 | On **independent** read-only calls, a parallel baseline matches or beats speculation; on **dependent** chains, speculation wins | Add a parallel-dispatch mode; run on tasks of both types | Crossover exists and is sharp |
| H12 | Speculation helps **p95 more than the mean** when tool latency has a heavy tail | Lognormal latencies with a 1-3% slow tail; compare mean vs p95 speedup | Larger p95 gain |
| H13 | Gain is **concave in chain depth**: early steps help most because later guesses compound errors | Vary chain length 2 to 8 on synthetic tasks; plot hit rate by step index | Hit rate decays with depth |
| H14 | A **cost-aware gate** (launch only if `p x saved > (1-p) x cost`) beats a fixed confidence threshold on the time-vs-waste frontier | Give tools different costs; compare Pareto curves | Dominates fixed threshold when costs vary |
| H15 | **Top-k speculation** (k=2 or 3 guesses) raises hits more than it raises waste, up to a concurrency limit | k sweep with a worker cap | Knee near k=2 or 3 |
| H16 | An **online-learning** speculator improves within a session and across sessions | Plot hit rate vs step number and vs session number | Rising curve on templated tasks, flat on vague ones |
| H17 | **Task-prefix speculation** (guess the first call from the prompt alone) is predictable on templated tasks | Remove the empty-history early return; score first-call hit rate | High on templated, near zero on vague |
| H18 | A **macro tool** captures most of speculation's savings on fixed chains, with no waste | Add a composite tool; compare time and waste | Wins on fixed chains, loses on flexible ones |
| H19 | Speculation **hurts** when speculator latency `s` exceeds tool time `D` | Sweep `s` against `D` (this refines H2) | Net-negative region exists |
| H20 | **Tool-name accuracy is much higher than full-call accuracy** for an LLM speculator, and a substring check on arguments closes part of the gap | Per-step logging of name-match vs exact-match | About 3x gap, narrowed by the check |
| H21 | **Staleness grows with TTL**: stale-hit rate rises as time between speculation and use grows, when mock data drifts | Time-varying price tool; vary delay | Rises with delay; scope-versioning removes it |
| H22 | **Reversible-write speculation** (hold, then commit) saves time with zero net side effects after rollback | Run prepare/rollback on booking tasks; compare final world state against baseline | Same final state; savings only on the commit step |
| H23 | **A speculator that sees the actor's model name and style predicts better** than a generic one | Same speculator, with and without "the actor tends to X" in its prompt | Small gain, unclear |
| H24 | Speculation benefit **vanishes when tools are fast** (below about 0.2 s) | `--scale` sweep | Matches H2 and the blog's negative result |

H11, H14, H15 and H22 are the ones I find most interesting: each changes a design decision, not just a number.

---

## 3. Different ways to take the whole project

Six project shapes. They share the harness. They differ in the question they ask.

### A. "The map" (current plan)
**Question:** When does exact-match, read-only speculation pay, and when does it lose?
**Deliverable:** break-even heatmap, oracle-gap per speculator, safety results.
**Pros:** Cleanly falsifiable, uses what's built, easy to finish.
**Cons:** Less novel on its own. The claim "speculation helps" is already published.
**Effort:** medium. **Risk:** low.

### B. "Do you even need speculation?" (head-to-head benchmark)
**Question:** On the same tasks and tools, how do speculation, parallel dispatch, macro tools, plan-then-execute, and batching compare on time, cost and correctness?
**Why it's unique:** The papers each compare to sequential execution, not to each other. A single harness running levers 1, 2, 3, 4 and 5 side by side is a real contribution *(inferred: I did not find a paper that does this, but I didn't search exhaustively)*.
**Deliverable:** a decision chart: "if your task looks like X, use lever Y."
**Pros:** Very practical, teaches the most about agent design.
**Cons:** More implementation (the mock agent needs a plan-execute mode).
**Effort:** high. **Risk:** medium.

### C. "Speculation as a decision problem" (cost-aware policy)
**Question:** Can we replace the fixed confidence threshold with an expected-value rule that accounts for tool cost, quota, and time saved?
**Pieces:** calibrate confidence, assign cost to each tool, add a budget, compare to a fixed threshold (H14, H15).
**Pros:** Clean math, a real design contribution, small code.
**Cons:** Needs good calibration data, which means live LLM runs.
**Effort:** medium. **Risk:** medium.

### D. "Speculation as an attack surface" (security spin)
**Question:** What can an attacker do by poisoning what the speculator reads, and which issue-time defenses stop it?
**Pieces:** injection canary, adversarial speculator, provenance and allowlist policies, stale-cache attack, exfiltration via discarded fetches. Builds on what the last commit started.
**Why it's unique:** The research report says no paper studies this directly, and "Ghost Tool Calls" is only seen at abstract level.
**Pros:** Memorable demo (a canary URL gets fetched by a naive speculator, then blocked). Strong story.
**Cons:** Less about speed, so the headline chart differs.
**Effort:** medium. **Risk:** low.

### E. "Speculative writes" (reversible-action spin)
**Question:** Which side-effecting actions can be started early safely using a two-phase pattern (hold, then commit; draft, then send; reserve, then confirm)?
**Pieces:** build a taxonomy of tools by reversibility, extend `prepare` and `undo`, measure savings against the cost of rollbacks, check world-state equality.
**Why it's unique:** The papers refuse writes entirely. The repo already has the first version (`hold_fare`, rollback).
**Pros:** Genuinely novel angle, builds on existing code.
**Cons:** Real-world reversibility is messy; the mock may flatter it. Must say so.
**Effort:** medium. **Risk:** medium.

### F. "Learn it from traces" (replay study)
**Question:** How predictable are next tool calls in real agent trajectories?
**Pieces:** load tau-bench trajectories, score top-1 and top-3 accuracy of rules, n-gram and LLM speculators on every prefix, then project speedup through the latency model.
**Pros:** Real data, cheap, no tool execution.
**Cons:** Measures prediction only. It cannot show actual speedup. License and format work needed.
**Effort:** medium. **Risk:** medium (the dataset read/write classification is unverified).

### G. Real-tools case study (validation spin)
**Question:** Do the mock-tool conclusions hold with real tools (a real search API, a real filesystem or GitHub API)?
**Pros:** Answers the main objection to a mock-tool project.
**Cons:** Cost, rate limits, noisy timing, keys to manage. Best as a small appendix, not the core.
**Effort:** medium. **Risk:** high.

### Comparison

| Shape | Novelty | Effort | Risk | Needs live LLM | Best demo |
|---|---|---|---|---|---|
| A. Map | Low-medium | Medium | Low | Optional | Heatmap |
| B. Head-to-head | High | High | Medium | No | Decision chart |
| C. Cost-aware policy | Medium | Medium | Medium | Yes | Pareto curve |
| D. Attack surface | High | Medium | Low | No | Canary live demo |
| E. Speculative writes | High | Medium | Medium | No | Booking hold-and-rollback |
| F. Replay study | Medium | Medium | Medium | Optional | Accuracy table |
| G. Real tools | Medium | Medium | High | Optional | Side-by-side timings |

---

## 4. Combos that fit in 16 days

These are my suggestions *(inferred)*, ordered by how well I think they fit the deadline.

1. **A + D (recommended).** The map is the spine, and the security work is the memorable second half. Most of D's pieces are already half built, and the plan's Phase 3 is basically D.
2. **A + E.** The map plus speculative writes. Strong if you want a novelty angle that isn't security.
3. **B, trimmed.** Compare only speculation, parallel dispatch and one macro tool. Three levers instead of five, so it fits. Pick this if you want the most transferable lesson.
4. **A + C.** The map plus a cost-aware gate. Pick this if you like the math side and have budget for live runs.

Avoid combining more than two shapes. The cut order in `PLAN.md` already says what to drop.

---

## 5. How to choose together

Answer these separately, then compare:

1. Which result would you be proudest to present?
2. Which would you be least embarrassed to report if it came out negative?
3. Which teaches you the most about how agents work?
4. Which can we finish with a clean, reproducible figure by Nov 29?
5. How much money and API time are we willing to spend?

If you disagree, the tiebreaker is that A is the spine of everything, so start with A's Phase 0-1 work either way. Both of you will learn the same foundations, and the pivot decision can wait until after the sweeps (Nov 20), when real numbers show which direction looks richest.
