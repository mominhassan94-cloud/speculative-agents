# Evaluation methodology, correctness guarantees, and safety pitfalls for speculative tool execution in LLM agents

(As of 2026-11-10. "Cited Findings" = what sources state. "Inferences" = my reasoning. Several 2026 papers were seen only via search-result summaries, not full text; flagged where relevant.)

## 1. Metrics used in the literature, and standard plots

### Takeaway
The two anchor papers (Speculative Actions, PASTE) both use end-to-end latency/time saved plus a prediction-accuracy/hit-rate metric; neither reports precision, wasted-call counts, or dollar cost for the lossless setting in a comparable way, and neither gives usable variance. A credible class project can beat both on reporting completeness by adding the missing metrics.

### Cited Findings
- PASTE defines E2E latency as "request arrival to final agent response", and also reports observed tool latency, p95/p99 tail latency, tool stall time, LLM queueing time, and throughput. Reported: E2E latency up to 43.5% lower on average, p99 up to 55.4% lower; tool latency up to 55.2% lower. — [PASTE arXiv 2603.18897](https://arxiv.org/html/2603.18897)
- PASTE "speculative hit rate" = probability that *any* speculatively executed prediction matches the next tool actually invoked; it also reports Top-1 accuracy (up to 27.8%), Top-3 recall (43.9%), overall hit rate 93.8%, and resource overhead (~1-3 idle CPU cores, ~250 MB memory; predictor+scheduler under 100 ms). The paper notes hit rate and Top-1 are not directly comparable. It does not report wasted/extra calls (beyond a side-effect count), precision, or monetary cost. — [PASTE](https://arxiv.org/html/2603.18897)
- PASTE ablations: PASTE-Tool-Only (no co-scheduler) and PASTE-LLM-Only (no speculative tools): E2E 419 s and 342 s vs 270 s full. Baselines: vLLM (no speculation), Agentix, ORION, SpecFaaS. — [PASTE](https://arxiv.org/html/2603.18897)
- Speculative Actions metrics: next-action prediction accuracy (chess: any of k speculative moves matches; e-commerce: fraction of speculative API calls matching ground truth; HotpotQA: exact (API, params) match), time saved = (T_seq - T_s)/T_seq, token overhead (reported for chess), cost in tokens/USD (OS tuning), p95 latency and convergence time (OS tuning). — [Speculative Actions arXiv 2510.04371](https://arxiv.org/html/2510.04371v1)
- Speculative Actions results: chess k=3 gives 19.5% average time saved at 54.7% accuracy; tau-bench retail 22-38% API prediction accuracy; HotpotQA up to 46% top-3 accuracy; abstract headline "up to 20% end-to-end lossless speedup". — [arXiv 2510.04371](https://arxiv.org/html/2510.04371v1); [search summary](https://arxiv.org/abs/2510.04371)
- Speculative Actions gives an analytical bound (Proposition 1): expected runtime ratio converges to 1 - (p/(1+p))(alpha/(alpha+beta)), at most ~50% reduction under exponential-latency and independence assumptions (p = per-step hit probability). — [arXiv 2510.04371](https://arxiv.org/html/2510.04371v1)
- Plots seen in these papers: time saved / token overhead vs. number of predictions k and steps (chess); accuracy vs. speculator model and reasoning budget, and price/latency vs. accuracy (e-commerce); latency/cost vs. convergence time (OS tuning); time-breakdown bars and concurrency scaling (PASTE). — [arXiv 2510.04371](https://arxiv.org/html/2510.04371v1); [PASTE](https://arxiv.org/html/2603.18897)
- Newer related work exists (AOSpec, SpecBox, Speculative Interaction Agents, Atomix, Cordon) but I could not extract their metric definitions (AOSpec PDF unreadable via fetch). — [AOSpec](https://arxiv.org/pdf/2608.00881)

### Inferences
- Metric set recommended for us: (a) E2E wall time, mean/median/p95 over runs, plus speedup = T_base/T_mode and % reduction, paired per (task, rep); (b) speculation quality: launched, hits, precision = hits/launched, recall = hits/real read-only calls, wasted = launched - hits; (c) cost: extra tool calls, extra billed tool-seconds/API cost units, extra speculator LLM tokens/USD; (d) "tool-wait fraction" = time blocked on tools / total; (e) correctness: task success rate and trajectory/answer equivalence vs. baseline; (f) safety counters: side-effect tools executed early (must be 0), blocked, held-back.
- Standard plots to make: speedup-vs-cost Pareto (x = extra calls or $ overhead, y = % latency saved, one point per mode/threshold/k); hit rate and precision vs. confidence threshold (plus latency saved on a second axis); latency vs. tool-latency scale (shows when speculation stops paying); per-task paired speedup bars with CI; box/violin of latency per mode for tail behavior; oracle-gap plot (achieved fraction of oracle savings).
- Because PASTE's "hit rate" counts any-of-many candidates while Speculative Actions counts per-guess matches, we should define ours explicitly and report both per-launch precision and per-step "any-hit".

### Gaps
- No source found defining "goodput" for speculative agents specifically; I treat it as an inference (useful work completed per unit of total work, i.e., committed-result time over total tool time incl. wasted).
- No comparable published dollar-cost-per-task numbers for tool-level speculation in the lossless setting.

## 2. Baselines and ablations

### Takeaway
Papers use no-speculation as the baseline, ablate the speculator (model, k, reasoning budget) and the scheduler; an oracle and always-speculate are standard-by-reasoning but not explicitly named in the sources I could read. LLMCompiler is the relevant parallel-calling comparison but solves a different problem (planner-extracted dependencies, not guessing).

### Cited Findings
- Speculative Actions baseline: sequential execution; ablations over k (3 main, k=1 vs 3 in HotpotQA), speculator model and reasoning budget (gpt-5-nano/mini/gpt-5, gemini-2.5-flash; budgets 1024/2048/4096), single- vs multi-model speculators. OS-tuning experiment uses Actor-only, Speculator-only, combined. — [arXiv 2510.04371](https://arxiv.org/html/2510.04371v1)
- Speculative Actions uses strict exact matching; it counts synonyms/word-order differences as misses and notes this penalizes stronger models; semantic matching is not evaluated. — [arXiv 2510.04371](https://arxiv.org/html/2510.04371v1)
- PASTE baselines vLLM, Agentix, ORION, SpecFaaS; hardware, LLM config, tool interfaces, timeouts and retry policy held constant across systems; predictor trained on historical tasks and evaluated on disjoint tasks. — [PASTE](https://arxiv.org/html/2603.18897)
- LLMCompiler (planner + task-fetching unit + parallel executor): up to 3.7x latency speedup, 6.7x cost saving, ~9% accuracy gain vs ReAct (abstract); body lists smaller figures (2.27x, 4.65x); v1 abstract: up to 1.35x vs OpenAI parallel function calling. Abstract and body numbers differ and I could not reconcile them. — [LLMCompiler arXiv 2312.04511](https://arxiv.org/abs/2312.04511)

### Inferences
- Baseline ladder for our harness: (0) baseline (no spec); (1) random/shuffled-guess speculator (lower bound / sanity); (2) always-speculate (threshold 0, read-only gate on) to show cost of unthresholded speculation; (3) rules; (4) LLM speculator (small model) once available; (5) oracle (upper bound; already present); (6) LLMCompiler-style parallel batch: only fair when the plan's independent calls are known up front; implement as "issue all independent read-only calls in the same step concurrently" and compare, since it is a non-speculative alternative that is also lossless.
- Ablations: confidence threshold sweep (0..1); k (top-k guesses); max concurrent speculative calls / budget; exact vs. normalized argument matching; speculator accuracy injected artificially (oracle with p-corrupted guesses, p in {0.25,0.5,0.75,1}) to map speedup vs. hit rate and compare against the Proposition 1 curve; tool latency scale; think-time scale (speedup depends on the ratio of LLM step time to tool time); safety gate on/off (in a sandboxed mock only) to show what the gate prevents.

### Gaps
- Sources I read do not explicitly name "always-speculate", "random", or "oracle" baselines; these come from my design reasoning.

## 3. Statistical practice for noisy LLM agents

### Takeaway
Agent outcomes are noisy even at temperature 0; use many repetitions, paired designs, and report CIs. The anchor speculation papers give almost no variance reporting, so reporting it ourselves is a differentiator. Note our current harness uses a deterministic scripted brain, so outcome noise there is only timing noise.

### Cited Findings
- A study of 60,000 SWE-Bench-Verified trajectories (10 runs per configuration): single-run pass@1 ranges span 2.2-6.0 points; std devs 0.7%-1.8%; temperature 0 does not give determinism (variance "never decreases and sometimes increases"); recommends multiple independent runs, power analysis, and reporting pass@k and pass^k. — [arXiv 2602.07150](https://arxiv.org/html/2602.07150v2)
- Same paper's power analysis (80% power, alpha 0.05): ~9 runs per agent to detect a 2-point gain at sigma=1.5%; ~36 runs for a 1-point gain; ~8 runs for 1 point at sigma=0.7%; ~10-point gains may be detectable with one run. It does not discuss bootstrap or paired tests. — [arXiv 2602.07150](https://arxiv.org/html/2602.07150v2)
- tau-bench introduced pass^k (succeed on all k attempts); reported example GPT-4o 61% pass@1 vs 25% pass@8 on retail (as summarized in [arXiv 2603.29231](https://arxiv.org/html/2603.29231v1) via search; not verified at the tau-bench source).
- Speculative Actions is inconsistent on run counts (text says 5 runs for chess, figure captions say 55) and reports no seeds; PASTE states no run counts, variance, or CIs. — [arXiv 2510.04371](https://arxiv.org/html/2510.04371v1); [PASTE](https://arxiv.org/html/2603.18897) (both per my fetch summaries)
- A vendor blog claims only 16% of 445 reviewed benchmarks ran any statistical test; secondary, non-peer-reviewed. — [LayerLens](https://layerlens.ai/blog/the-test-your-agent-never-took)

### Inferences
- Paired design: run baseline and each mode on the same (task, rep) with the same seeds/latency jitter draw; analyze per-pair ratio T_mode/T_base (log scale) and report geometric-mean speedup with a bootstrap 95% CI (resample tasks, then reps within tasks). Use Wilcoxon signed-rank or paired t on log-ratios. Order effects: randomize or interleave mode order (current harness runs baseline, rules, oracle in fixed order per rep, so warm caches/threads could bias).
- Reps: for latency with scaled sleeps, timing noise is small; 10-30 reps per (task, mode) is cheap. For real-LLM agents, use >=10 reps per task (matching the 2602.07150 practice) and report success as pass@1 mean +/- CI and pass^k for the "lossless" claim.
- With real LLMs, condition on trajectory: compare speculation vs. baseline either (a) by replaying recorded baseline trajectories (isolates speculation effect; makes "lossless" exactly testable because the agent's observations are identical) or (b) end-to-end live (captures real nondeterminism). Report both.

### Gaps
- No source found on bootstrap CI methods specific to pass^k or to latency ratios; the recommendations above are standard statistics, not from a cited agent paper.

## 4. Defining and testing "lossless"

### Takeaway
"Lossless" in the literature means the agent's outcome/trajectory is unchanged versus sequential execution, enforced by buffering speculative results outside authoritative state and committing only on a matching real call; guarantees are argued by design, tested mostly by outcome comparison.

### Cited Findings
- Speculative Actions: runs should match sequential outcomes (chess trajectory "remains identical to non-speculative play"); guards = semantic-equivalence check before commit, safety envelopes limited to idempotent/reversible/sandboxed effects, and rollback/compensating repair. The OS-tuning case is explicitly *lossy* (last-write-wins). — [arXiv 2510.04371](https://arxiv.org/html/2510.04371v1)
- PASTE: speculative results stay outside authoritative session state until the LLM emits a matching call; only reuse/promotion commits. "A wrong prediction can waste bounded work, but it cannot expose an unconfirmed result to the agent." Audit of >20,000 speculative actions: 602 potentially side-effecting, prevented from committing, and no task's final result differed from baselines. Per-tool breakdown not given. — [PASTE](https://arxiv.org/html/2603.18897)

### Inferences
- Define three levels and test each: (L1) *outcome-lossless*: final answer correct/equivalent to baseline (current harness `correct` flag; weakest). (L2) *trajectory-lossless*: the sequence of real (non-speculative) tool calls and the observations the agent receives are identical to baseline's, given identical agent decisions; test by logging the agent-visible transcript and diffing/hashing across modes. (L3) *world-state-lossless*: external state (mock DB, booking ledger, send log, rate-limit counters, audit log) after the run is identical to baseline's, including no extra side effects; test by snapshotting mock state.
- Staleness must be part of the definition: a cache hit is lossless only if the returned value equals what a fresh call at use-time would return. Test: for each hit, optionally re-execute the tool at use time (shadow check) and compare; report "stale-hit rate".
- Fault-injection tests: (i) mutate a mock resource between speculation and use (see Section 5) and assert the stale result is not served; (ii) tool errors/timeouts in speculative calls must not surface; (iii) a wrong guess must leave no trace.
- Current harness only checks `correct` (answer string) and `side_effect_early`; it does not check L2/L3 or stale hits.

### Gaps
- No standard published test suite for "lossless" checking; PASTE's 602-case audit methodology is not described in detail.

## 5. Correctness hazards of speculating "read-only" tools

### Takeaway
Read-only does not mean safe to run early: time-sensitivity, rate limits, per-call billing, audit logging, access control, and attacker-controlled content all break the equivalence between speculative and real calls. Sources document the side-effect/rollback side well; most of the specific hazards below (staleness, billing, rate limits, logging) are not examined in the speculation papers and are my inference.

### Cited Findings
- Speculative Actions assumes pre-launched calls have no visible side effects or are rollback-able; it does not explain how write-API side effects are handled in the e-commerce and HotpotQA experiments. — [arXiv 2510.04371](https://arxiv.org/html/2510.04371v1)
- PASTE: tools with side effects need an explicit safe-speculation policy or a non-mutating variant (dry-run / staging); otherwise not speculated. Partial predictions are limited to preparation work. Authoritative calls keep priority; speculative work is preempted first. No rollback mechanism described; isolation is used. Sandboxing mechanism not described. — [PASTE](https://arxiv.org/html/2603.18897)
- Atomix (per search summary): cancellations are not atomic or guaranteed, so losing speculative branches can leave residual side effects or temporary inconsistency; it buffers delayable effects and compensates externalized ones on abort. Cordon: snapshot rollback cannot retract a dispatched external effect. SpecBox: prepares environments early but holds side-effecting work until commit; cache reuse limited to deterministic, compatible requests. Speculative Interaction Agents: read-only tools marked safe, state-changing tools held until confirmation. — [Atomix](https://arxiv.org/html/2602.14849v1); [Cordon](https://arxiv.org/pdf/2606.17573); [SpecBox](https://arxiv.org/pdf/2607.23933); [Speculative Interaction Agents](https://arxiv.org/pdf/2605.13360) (all via search summary only, not full text)
- A Claude Code feature request for speculative tool-call sequences proposes each speculative call still pass permissions/hooks/sandbox and lists billing for discarded speculative tokens as an unresolved question. — [claudeissues.com #96404](https://claudeissues.com/issue/96404-feature-speculative-tool-call-sequences-with-expected-results-branch-prediction)
- MCP spec: servers MUST validate inputs, implement access controls, rate limit, sanitize outputs; clients SHOULD prompt for confirmation on sensitive operations, validate tool results before passing to the LLM, implement timeouts, and log tool usage for audit; and there SHOULD be a human in the loop able to deny tool invocations. — [MCP spec 2025-06-18, Tools](https://modelcontextprotocol.io/specification/2025-06-18/server/tools)

### Inferences (hazard checklist for our design)
- Stale cache / time sensitivity: a cached result computed at t0 and used at t1 may differ (prices, availability, inventory, weather). Need TTL per tool and invalidation on any intervening write, plus a staleness check at use time.
- Read-after-write ordering: if a side-effect tool runs between speculation and use, any speculative result that depends on the mutated resource is invalid even with exact argument match. Fix: tag each tool with the resource/"state scope" it reads and writes; keep a version counter per scope; discard cache entries whose scope version changed (or flush cache on any write).
- Rate-limit consumption and per-call billing: speculative calls consume quotas and money even when discarded; they can also cause throttling of the real call. Track as cost metric and add a speculation budget.
- Logging/audit side effects: a "read" may write access logs, mark-as-read, update "last viewed", trigger analytics, or alert; wasted speculative reads pollute audit trails and could be misread as user intent.
- Authorization/privacy: a speculative call must run with no more privilege than the real call and only on data the user would be allowed (and plausibly would choose) to access; speculative reads of sensitive records the agent would not have opened leak access/audit exposure even if the result is discarded. Per-user cache keys needed to avoid cross-user cache leakage.
- Race conditions: speculative call in flight when the real call arrives (join the in-flight future vs. duplicate); speculative call finishing after the agent moved on; cancellation not guaranteeing effect-free abort (Atomix point).
- Non-idempotent "reads": e.g., pagination cursors, one-time tokens, queue pops, "get next" APIs. These look read-only but mutate state.
- Exact-match risk: normalization bugs (whitespace, defaults, ordering) could cause a hit on semantically different args; or too strict matching lowers hit rate (the Speculative Actions observation).

### Gaps
- The speculation papers do not quantify staleness, rate-limit, billing, or audit-log effects; I found no empirical study of these for speculative tool execution.

## 6. Classifying tools as speculation-safe (annotations, sandboxes, rollback)

### Takeaway
MCP provides hint-only annotations (readOnlyHint, idempotentHint, etc.) that the spec says must be treated as untrusted unless from a trusted server, with defaults that are pessimistic. Papers use manual/policy classification plus sandboxing or buffering; none relies solely on self-declared annotations.

### Cited Findings
- MCP ToolAnnotations: all properties are hints; clients must treat annotations as untrusted unless from trusted servers; "Clients should never make tool use decisions based on ToolAnnotations received from untrusted servers." Defaults: readOnlyHint=false; destructiveHint=true (meaningful only when readOnlyHint is false); idempotentHint=false (meaningful only when readOnlyHint is false); openWorldHint=true. — [MCP spec Tools](https://modelcontextprotocol.io/specification/2025-06-18/server/tools); defaults per WebFetch summary of the MCP schema.ts at [github.com/modelcontextprotocol](https://raw.githubusercontent.com/modelcontextprotocol/modelcontextprotocol/main/schema/2025-06-18/schema.ts)
- PASTE: safety classes = side-effect-free and fully parameterized (run end to end), partial predictions (preparation only), side-effecting with explicit policy or dry-run variant, otherwise never. — [PASTE](https://arxiv.org/html/2603.18897)
- Speculative Actions: "safety envelopes" of idempotent, reversible, or sandboxed effects with rollback/compensation. — [arXiv 2510.04371](https://arxiv.org/html/2510.04371v1)
- AOSpec (per search summary): sandboxes such as CubeSandbox and DeltaBox isolate and revert file/process effects, enabling speculation over side-effecting tools. — [AOSpec](https://arxiv.org/pdf/2608.00881)

### Inferences
- A speculation-safe predicate should require: readOnlyHint true AND (for network tools) openWorldHint reviewed AND declared by us/trusted server AND no per-call-billing/quota flag AND no one-time-token semantics AND tool has a TTL. Treat missing annotations as unsafe (matches MCP defaults). Note idempotentHint alone is not sufficient for speculation: an idempotent write still has visible effects (e.g., idempotent "create order with key").
- Our mock_tools `read_only` boolean maps to readOnlyHint; add fields for `ttl_s`, `cost_per_call`, `state_scope`, `open_world`, `rate_limited` so the harness can test the hazards in Section 5.

### Gaps
- No empirical evaluation found of how accurate real-world MCP annotations are (i.e., how often readOnlyHint is wrong).

## 7. Security research affecting early execution

### Takeaway
Indirect prompt injection via tool results is the main documented channel; speculation adds a new angle (fetching attacker-influenced content before the agent or user chose to), but I found no paper analyzing that specifically. The safe design principle already in the speculation papers (do not expose unconfirmed results to the agent) is also the main mitigation.

### Cited Findings
- InjecAgent: 1,054 test cases, 17 user tools, 62 attacker tools; GPT-4-based agent attack success rate 24%, 47% with a "hacking prompt". AgentDojo is the other standard benchmark. — [InjecAgent arXiv 2403.02691](https://arxiv.org/html/2403.02691v1); summary via [search](https://arxiv.org/html/2510.05244v1)
- Bhagwatkar et al. report flawed success metrics and bugs in AgentDojo and Agent Security Bench and propose a tool-output sanitizer firewall; a 2026 paper reports sanitizing only the first tool observation captures 67% of measured injection successes. (Search-summary level; not verified in full text.) — [arXiv 2510.05244](https://arxiv.org/abs/2510.05244v2)
- OWASP LLM Top 10 (2026): prompt injection stays #1; excessive agency rose from 6th to 3rd; mitigations include tightly scoped permissions and monitoring. Sources disagree on publication date (2026-08-03 vs 2026-08-15); verify at the official OWASP GenAI site. — [SD Times](https://sdtimes.com/security/prompt-injection-tops-2026-owasp-genai-llm-top-ten-vulnerabilities/); [Check Point](https://blog.checkpoint.com/ai-security/reading-the-signals-in-the-owasp-llm-top-10-2026/)
- MCP: clients SHOULD validate tool results before passing them to the LLM and treat annotations from untrusted servers as untrusted. — [MCP spec Tools](https://modelcontextprotocol.io/specification/2025-06-18/server/tools)

### Inferences
- Speculative fetch of attacker-controlled pages: even if the result is discarded, the *fetch itself* can be an exfiltration channel (URL/query carrying data, tracking pixel, server-side callback) or trigger server-side effects (GET with side effects, e.g., "unsubscribe" links). A prompt-injected speculator input (speculator reads the previous tool result, which may be attacker-controlled) could be steered to guess attacker-chosen URLs, so speculation extends the attack surface to the speculator, not just the main agent. Mitigations: speculator sees sanitized/truncated observations; allowlist of domains/tools and argument provenance (only speculate on args that appear verbatim in prior trusted outputs or user prompt, never attacker-synthesized URLs); no speculation for open-world tools on untrusted origins.
- Cache poisoning: a speculative result cached under (tool,args) and reused by a later real call imports whatever content was returned; keep cache per-session, per-user, short-lived, and never share across sessions; hit only for the exact real call, which preserves baseline exposure (the agent would have seen the same page anyway) but not the timing/ordering of that exposure.
- Evaluation: add an injection test where a mock search result contains a link to a canary URL and an instruction; measure (a) whether speculation fetched the canary (speculative-exposure count), (b) whether the agent ever saw it unless it requested it. AgentDojo/InjecAgent are the references for baselines but are much larger than a class project needs.

### Gaps
- No paper found that studies prompt injection or exfiltration specifically through speculative/prefetched tool calls.
- I did not read the full OWASP documents, only secondary summaries.

## 8. Concrete recommendations for our harness (run_eval.py)

### Takeaway
The harness already has baseline/rules/oracle, scaled latency, side-effect counters and CSV output; the main gaps are statistical rigor, cost/precision metrics, threshold and accuracy sweeps, paired analysis, and hazard (stale/ordering/injection) tests.

### Cited Findings
- Facts about the current repo (read from local files): run_eval.py runs modes baseline/rules/oracle with `--reps` (default 3), `--scale` (default 0.1), `--think`; per-run CSV columns: task, group, mode, seconds, correct, answer, launched, hits, wasted, blocked, held_back, est_saved_s, side_effect_early, rep; brain is a deterministic scripted PlanBrain; mode order within a rep is fixed baseline->rules->oracle; summary prints means only (no CI); oracle deliberately guesses book/send to show the gate blocks them. — local: /Users/mominhassan/speculative-agents/run_eval.py

### Inferences (prioritized to-do list)
1. Statistics: raise default reps to >=20 for scripted runs (>=10 for LLM runs); randomize mode order per rep; add `summarize` output of median, p95, geometric-mean speedup with bootstrap 95% CI, and paired per-(task,rep) ratios; write a separate analysis script/notebook that reads the CSV and makes the plots.
2. Metrics columns: add precision (hits/launched), recall, wasted_tool_seconds, extra_calls, simulated_cost (add `cost` to mock tools and speculator token counts), tool_wait_s, speculator_overhead_s; log per-call events to a second CSV/JSONL (tool, args, spec/real, start, end, hit, discarded reason).
3. Sweeps: `--threshold` sweep (0.0-1.0) for rules/LLM speculator; `--k`; `--scale` sweep; `--think` sweep; an `accuracy` mode = oracle with each guess corrupted with probability 1-p (compare measured speedup to the Proposition 1 shape).
4. New baselines: `random` speculator, `always` (threshold 0), `parallel` (LLMCompiler-style concurrent independent calls), and later an `llm` speculator; keep oracle as the upper bound and report "fraction of oracle savings captured".
5. Lossless tests: log the agent-visible transcript per run and assert identical across modes (L2); snapshot mock world state and assert identical (L3); keep `side_effect_early` but also count speculative calls to any tool not marked safe; optional shadow re-execution to detect stale hits.
6. Hazard scenarios as dedicated tasks: (a) write-between-speculate-and-use (e.g., search stock, then a mutation tool changes it, then real call); (b) time-varying tool (price changes with a TTL); (c) rate-limited/billed tool with counter; (d) read tool that appends to an audit log; (e) injection canary in a search result; (f) non-idempotent "read" (next-page token). Expected outcomes: zero stale hits served, zero early side effects, counters reported in the CSV.
7. Reporting for a class project: state seeds and config, include the oracle and baseline in every plot, show the failure/negative cases (speculation that hurts at low accuracy or low tool latency), and explicitly list threats to validity (mock tools, scripted brain, no real LLM nondeterminism, small task set). Optionally run a small real-LLM replication (a few tasks x >=10 reps) to show noise.

### Gaps
- Not examined: the contents of speculator.py/runtime.py beyond file sizes and README claims, so recommendations about cache keys/TTL are not checked against the current implementation.
