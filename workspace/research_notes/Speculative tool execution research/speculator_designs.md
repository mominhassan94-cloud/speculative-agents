# Speculator (next-action predictor) designs for agent loops

Research date: 2026-11-10. Sources are mostly 2025-2026 arXiv preprints; most are not independently replicated. "Cited Findings" are paper-reported (some via a fetch tool's summary of the paper, so verify exact figures before quoting); "Inferences" are mine.

## 1. Small-LLM prompting speculators (Haiku/Flash class), calibration, argument copying

### Takeaway
A small LLM given the same context as the actor predicts the exact tool call only 20-45% of the time (top-1 to top-3) on open-ended tasks, and the hit rate is far higher for the tool name than for exact arguments. Closed APIs (Claude) do not expose logprobs, so confidence must come from verbalized scores or self-consistency across samples. I found no head-to-head calibration study inside a tool-calling loop.

### Cited Findings
- Speculative Actions (Columbia, ICLR 2026): a fast speculator and a slow actor see the same context. Speculators included gpt-5-nano/mini, gemini-2.5-flash, gpt-4.1-nano. Reported next-action accuracy: chess 54.7% with 3 predictions; tau-bench retail 22-38% of API calls; HotpotQA up to 46% with top-3. About 20% end-to-end lossless speedup. — [arXiv 2510.04371](https://arxiv.org/html/2510.04371v1)
- In the same paper, stronger speculator models sometimes score lower on HotpotQA because their phrasing varies more, and HotpotQA used strict match (identical API name and parameters). — [arXiv 2510.04371](https://arxiv.org/html/2510.04371v1)
- Pooling multiple speculator models (nano+Gemini, mini+Gemini, gpt-5+Gemini) into one candidate set beat single models on tau-bench retail. Low-budget speculators respond in about 2-3 s. — [arXiv 2510.04371](https://arxiv.org/html/2510.04371v1)
- Practitioner experiment (GPT-5 actor, GPT-5-nano speculator, Jaccard 0.7 match, GAIA): hit rates only 18.2% (L1) and 10.5% (L2); average latency got worse (173.8s to 184.1s on L1; 198.5s to 264.3s on L2) and accuracy dropped (60.4 to 58.5; 53.5 to 48.8). On the subset where speculation helped: 63% (L1) and 37% (L2) speedup. Blog, single author, not peer reviewed. — [yogesh.bearblog.dev](https://yogesh.bearblog.dev/speculative-tool-calling/)
- Same blog: Co-Sight with 3 draft predictions executed on consensus went 169s to 113s; Open Deep Research went 128s to 162s (slower) because the draft proposed broader searches and tool-idle time was under 5%. — [yogesh.bearblog.dev](https://yogesh.bearblog.dev/speculative-tool-calling/)
- Same blog: draft model preferred a different search tool (Brave); when the draft chose DuckDuckGo the target chose a different tool 57% of the time. A LoRA-tuned Qwen3-4B (90 synthetic trajectories) lifted tool-name accuracy from 1.9% to 51.9% on a 4-tool set but fell to 31.5% on a 17-tool set with near-duplicate tools, over-predicting `final_answer` and `vision_ocr`. — [yogesh.bearblog.dev](https://yogesh.bearblog.dev/speculative-tool-calling/)
- Self-speculation (same model, not a smaller one) beats smaller same-family drafters: Qwen3-4B prompted to predict its own next call got 25.3 Hit@1 on MuSiQue vs 14.7 for the best external drafter, and 37.2 vs 25.4 on tau-bench. Best external drafter averaged 18.6 Hit@1. — [arXiv 2607.25816](https://arxiv.org/html/2607.25816)
- Calibration generally: token logprobs beat verbalized confidence for selection in agent settings (Soft-SC with logits beat verbalized by 2.4% at k=10 on Bash). — [Soft Self-Consistency, arXiv 2402.13212](https://arxiv.org/pdf/2402.13212)
- Verbal confidence carries information beyond token logprobs in Gemma 3 27B / Qwen 2.5 7B (QA settings, not agents). — [arXiv 2603.17839](https://arxiv.org/abs/2603.17839)
- Tool-use agents are miscalibrated in a tool-dependent way: evidence tools like web search induce overconfidence; verification tools like code interpreters reduce it. — [Confidence Dichotomy, arXiv 2601.07264](https://huggingface.co/papers/2601.07264)
- Opinion piece (not peer reviewed) says Claude does not expose token logprobs, suggesting self-consistency (sample N, measure agreement) as the substitute. — [DEV Community](https://dev.to/ji_ai/why-token-logprobs-beat-asking-your-llm-how-confident-it-is-4jk1)
- Structured output: Anthropic's fine-grained tool streaming streams tool-input JSON fragments without server-side validation, so partial/invalid JSON is possible; `eager_input_streaming: true` per tool. — [Anthropic docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/fine-grained-tool-streaming)

### Inferences
- Practical design for a Python student project: prompt Haiku/Flash with the same tool schemas, last N steps, and ask for top-3 JSON candidates (name + args) with a verbalized probability. Use constrained tool-call output (forced tool_use / JSON schema) so names are always valid. Calibrate the verbalized score post hoc on logged data (isotonic/Platt on hit/miss) rather than trusting it raw.
- Run k=3-5 samples at temperature and use agreement frequency as a second confidence signal; combine both in a small logistic regression trained on your own hit/miss log. This is my suggestion; no cited paper compares the three signals in an agent loop.
- Argument hallucination is the main failure: names are easy, exact args are hard (see Q3 SPORK numbers). Mitigate by instructing the speculator to copy argument values only from strings present in context (URLs, paths, ids), and by validating each predicted arg string is a substring of prior context; reject candidates containing unseen strings. (My inference, motivated by PASTE's 95% substring statistic below.)
- A speculator slower or less-correlated with the actor than the actor's own latency gives no benefit; same-family small models track the actor best, and self-speculation tracks it best of all.

### Gaps
- No published comparison of verbalized vs logprob vs self-consistency confidence specifically for next-tool-call hit prediction.
- No published numbers for Claude Haiku as speculator; the papers use GPT-5-nano/mini, Gemini Flash, Qwen.
- Prompt-format ablations (few-shot count, history length) for speculators were not found.

## 2. Pattern / n-gram / Markov mining over past trajectories (PASTE-style)

### Takeaway
PASTE-style systems key predictions on short event contexts (tool type plus success/failure) and fill arguments by late-bound dataflow mappers learned from history (e.g. first URL in the search result). Top-1 is modest (up to 27.8%) but top-3 recall is higher (43.9%), and gains come from budgeted speculation of cheap read-only tools.

### Cited Findings
- PASTE (Microsoft Research, arXiv Mar 2026): a "pattern" is a record of recent event context, future tool target, argument mapper, and empirical confidence. Context matches stable metadata such as tool type and success/failure. — [arXiv 2603.18897](https://arxiv.org/html/2603.18897)
- Mining: split events into a signature (control flow, excludes volatile NL) and payload (values that may flow to later arguments). Pass 1 mines recurring short contexts over signatures; pass 2 checks historical occurrences to infer simple mappers (field lookup in structured output, indexed result selection with fallback, string normalization); candidates validated on historical traces and kept only above an empirical probability. — [arXiv 2603.18897](https://arxiv.org/html/2603.18897)
- Dataflow example: after a successful search predict a web fetch with URL = first URL in the search result; after a failed fetch, a fallback pattern picks the next URL. Supporting stats: 55% of successful file_editor calls are followed by terminal execution; in 95% of web visits the URL is a strict substring of the preceding search output. — [arXiv 2603.18897](https://arxiv.org/html/2603.18897)
- PASTE accuracy: top-1 up to 27.8%, top-3 recall up to 43.9%, overall hit rate up to 93.8% (inflated because several candidates are speculated). Higher in structured loops (compile/test), lower in open-ended research. — [arXiv 2603.18897](https://arxiv.org/html/2603.18897)
- PASTE results: up to 43.5% lower average task completion time, 55.4% lower p99; tool latency down up to 55.2%; 1.71x over ORION and 1.83x over SpecFaaS tool speedup. (The abstract and a search summary state 48.5% / 1.8x; the numbers differ between the abstract and the fetched body text, so quote the paper's own version carefully.) — [arXiv 2603.18897](https://arxiv.org/html/2603.18897), [Microsoft Research page](https://www.microsoft.com/en-us/research/publication/act-while-thinking-accelerating-llm-agents-via-pattern-aware-speculative-tool-execution/)
- Overhead: predictor plus scheduler under 100 ms; 1-3 idle CPU cores and 250 MB for a 48% tool-latency reduction. — [arXiv 2603.18897](https://arxiv.org/html/2603.18897)
- AutoTool: k-th order Markov over tool sequences plus semantic similarity to pick next tool; reports up to 30% inference-cost reduction with competitive task success (no finer breakdown found). — [arXiv 2511.14650](https://arxiv.org/abs/2511.14650)
- TVCache: tool call graph with longest-prefix match; hit rates up to 70%, median tool execution up to 6.9x faster (result caching, not speculation). — [arXiv 2602.10986](https://arxiv.org/html/2602.10986v1)
- AWO mines recurring tool-call sequences into meta-tools. — [arXiv 2601.22037](https://arxiv.org/html/2601.22037v2)

### Inferences
- Student implementation: key = (last tool name, last status) and (last 2 tool names, status); store counts of next tool and of the argument "source" (which prior result field, regex group, or user-message span supplied each argument). Predict with back-off from order-2 to order-1 to unigram, emit top-k with empirical frequency as confidence.
- Dataflow templates can be learned cheaply: for each historical call argument value, search earlier results in the same trace for the value (exact, then normalized); record a path such as `results[-1].json.items[0].url`. Keep a template if it reproduces the argument in >= X% of occurrences (X ~ 80-90% is my guess).
- Data needed: the PASTE excerpt did not state trace count or threshold. I expect tens to low hundreds of trajectories per task family for order-1/2 tool-name stats, and more for rare templates; this is untested.
- Pattern mining fits repetitive workflows (coding loops, fixed pipelines) and fails on open-ended research, where an LLM speculator is better; a hybrid (n-gram when confident, LLM otherwise) is a natural design.

### Gaps
- PASTE trace counts, confidence threshold, and per-benchmark breakdown were not in the portion I read (final ~1.2k chars of the page unread).
- No published learning-curve (accuracy vs number of training trajectories) for n-gram next-tool predictors found.
- AutoTool's per-metric numbers (LLM calls, tokens) not retrievable (PDF unreadable to the fetch tool).

## 3. Learned predictors and using the agent's own partial output

### Takeaway
Best evidence for "learned" predictors: self-speculation by the agent itself (SPORK, training-free; Joint RL, trained) and fork-at-start probes. Tool name is predictable early (74.6-99.6%) but exact arguments are not (7.6-83.9%), rising as chain-of-thought accumulates.

### Cited Findings
- SPORK (Aug 2026 preprint, code at github.com/baihuajun24/spork): after the main generation's first token, fork a probe with a forced tool-call prefix; dispatch the speculated tool when confidence reaches 0.90; reuse the result only on strict match of name and serialized arguments. — [arXiv 2607.03333](https://arxiv.org/html/2607.03333)
- SPORK tool-name accuracy at fork-at-start, Qwen3-32B: GAIA 98.3%, tau2 74.6%, BrowseComp 83.7%, HotpotQA 99.6%, BFCL 99.2%. Argument-exact 7.6-83.9%; on GAIA 7.6% at fork-at-start, 61.0% at 80% CoT, 97.5% at think-end. — [arXiv 2607.03333](https://arxiv.org/html/2607.03333)
- SPORK speedups: GAIA P95 131.9s to 108.1s (-18%), P50 -10%; tau2 mean 1.09x (0.5 s tools) to 1.18x (5 s tools); no-think mode 0.79x (slower); Qwen3-4B about neutral on GAIA/HotpotQA. Only read-only tools speculated. Closed APIs cannot host the fork. — [arXiv 2607.03333](https://arxiv.org/html/2607.03333)
- Speculate While You Reason (arXiv 2607.25816): one model with two modes sharing the prefix KV cache, joint agent-speculator RL; average Hit@1 Qwen3-4B 29.1 (base) to 44.1 (SFT) to 61.2 (RL); Qwen3.5-4B 33.2 to 48.9 to 66.3; task success not degraded. No end-to-end latency speedup reported; tested only at 4B scale. — [arXiv 2607.25816](https://arxiv.org/html/2607.25816)
- Fine-tuning a small model on very few trajectories helps with few tools but degrades with near-duplicate tools (see Q1 blog results). — [yogesh.bearblog.dev](https://yogesh.bearblog.dev/speculative-tool-calling/)
- Anthropic fine-grained tool streaming (`eager_input_streaming`) delivers input_json_delta fragments as generated; contract: init string on content_block_start, append partial_json, parse on content_block_stop; may be invalid JSON; available on all current Claude models; no beta header needed. — [Anthropic docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/fine-grained-tool-streaming)
- Practitioner/analysis note: tool name often decodes before arguments finish, so warm-up work can start then; partial-argument execution only works for tools tolerant of incomplete input. — surfaced via search of [yogesh.bearblog.dev](https://yogesh.bearblog.dev/speculative-tool-calling/) and related posts (secondary summary; not directly verified)
- Engine-side variant (client submits finished tool results to inference engine's tool cache; cached argument tokens injected as draft tokens); client-side speedup bounded below 2x. — [arXiv 2512.15834](https://arxiv.org/html/2512.15834)

### Inferences
- With the Claude API, a cheap "streaming early-dispatch" speculator is feasible: parse partial JSON with an incremental parser (e.g. `partial-json-parser` or manual), and as soon as the tool name and a complete first argument appear, launch the call if the tool is read-only and the required args are complete. This is lossless by construction (the args are the actor's own), so it avoids prediction error and only saves the tail of generation time; benefit scales with how long the remaining args/stop take (largest for tools with big payloads). Distinguish this from true prediction.
- Reuse of "think-then-act" structure: for models that reason before calling a tool, the pre-call text often states the intended tool and target; a small model reading the streamed reasoning prefix can predict the call (my extrapolation of SPORK's rising arg accuracy with CoT fraction, 7.6% to 97.5%, which used logprob probes on open weights).
- Fine-tuning route for a student: collect (context, next call) pairs from own logs, LoRA a 0.5-4B model, or train a tool-name classifier (e.g. gradient boosting / small transformer over last-k tool ids + embeddings) and a separate argument extractor via the dataflow templates. Retrieval variant: embed the current context, retrieve the nearest past trajectory step, copy its tool and template-fill its args. None of these was directly benchmarked in the sources I found.

### Gaps
- No paper found that benchmarks retrieval/nearest-trajectory prediction or a dedicated tool-name classifier against LLM speculators.
- No published data on predicting from Claude's partial streamed output specifically.
- SPORK's and 2607.25816's results are open-weight Qwen models; transfer to Claude/GPT-class is unverified.

## 4. Multi-guess strategies, budgets, thresholds

### Takeaway
Top-k with k around 3 is the common choice; hit rate rises materially from top-1 to top-3 but cost scales with k. Thresholding on confidence (0.90 in SPORK) cuts wasted speculation: precision 88% and acceptance up from ~0.22 to ~0.37.

### Cited Findings
- Speculative Actions: theoretical runtime ratio 1 - (p/(1+p))*(alpha/(alpha+beta)); at most 50% reduction (p = 1, alpha to infinity). Confidence-weighted and tree (multi-step) speculation proposed but not evaluated. — [arXiv 2510.04371](https://arxiv.org/html/2510.04371v1)
- Speculative Actions token cost rises with number of predictions (OS tuning example: Speculator-only 2.10M tokens/21.5 cents vs Actor-only 63k tokens/2.18 cents vs combined 608k tokens/7.83 cents). — [arXiv 2510.04371](https://arxiv.org/html/2510.04371v1)
- PASTE scheduler admission: executable, policy-safe, enough confidence and expected benefit, and budget headroom; candidates greedily ranked by expected utility per unit resource; speculative jobs bounded, low priority, preemptible; outcome per job: reused, promoted, discarded, preempted. Blind speculation (Tool-Only) was worse: 419 s vs 342 s LLM-Only in an ablation. — [arXiv 2603.18897](https://arxiv.org/html/2603.18897)
- SPORK gate: confidence = min top-1 token probability over the tool-name span; theta=0.90 gives 88% precision, 100% recall, F1 0.937 on 997 GAIA probes, filtering 77%; mean span probability 0.90 for correct vs 0.46 for incorrect. A cost model predicts speedup within 1.84% residual. — [arXiv 2607.03333](https://arxiv.org/html/2607.03333)
- Co-Sight integration executes speculation on consensus of 3 draft predictions. — [yogesh.bearblog.dev](https://yogesh.bearblog.dev/speculative-tool-calling/)

### Inferences
- Simple cost/benefit rule: launch candidate i if p_i * T_tool_saved > (1 - p_i) * C_waste + C_launch, where T_tool_saved is the latency hidden (bounded by actor think time), C_waste is dollar/compute cost converted at a chosen exchange rate. Learn p_i by calibrating confidence against logged hits, and learn the threshold by sweeping it offline over logged traces to maximize net saving.
- Per-tool statistics (mean latency, cost, side-effect class) should enter the rule: speculate cheap, read-only, slow tools; never speculate writes unless sandboxed.
- Cap concurrent speculative calls (budget k) and rank by p_i / cost_i (PASTE-like greedy).

### Gaps
- No paper reports a learned threshold optimized from cost/benefit end to end; thresholds in sources are hand-set (0.90) or ranked greedily.
- Optimal k vs cost curves for different tool latencies not found beyond the Speculative Actions analysis.

## 5. Argument matching: exact vs relaxed, "lossless"

### Takeaway
Papers that claim losslessness use strict (canonicalized) match of tool name plus all arguments; relaxed matching raises hit rate but can silently change behavior, so it is lossy in the strict sense.

### Cited Findings
- Speculative Actions: lossless = final outcome identical to sequential execution; relies on the actor validating each speculated action before commit, side effects reversible/sandboxed, and rollback on rejection. HotpotQA used strict match, which penalizes semantically equivalent calls. — [arXiv 2510.04371](https://arxiv.org/html/2510.04371v1)
- SPORK: strict match of name and serialized arguments, otherwise run serially. — [arXiv 2607.03333](https://arxiv.org/html/2607.03333)
- 2607.25816: Hit@1 is exact match on name and full argument dict; training reward uses partial credit (macro token-F1 over arg keys). — [arXiv 2607.25816](https://arxiv.org/html/2607.25816)
- Engine-side caching keys on tool name plus canonicalized arguments. — [arXiv 2512.15834](https://arxiv.org/html/2512.15834)
- PASTE canonicalizes invocations so later authoritative calls can be matched; 602 of 20,000+ speculative actions were side-effecting and blocked; no task changed its final result. — [arXiv 2603.18897](https://arxiv.org/html/2603.18897)
- Relaxed matching experiments (blog): strict, name-only, embedding cosine. HotPotQA with cosine 0.8: 81% cache rate, 76% accuracy; tool-name-only reuse can use results produced with wrong arguments; cosine/Jaccard thresholds are tuning choices trading hit rate against correctness. — [yogesh.bearblog.dev](https://yogesh.bearblog.dev/speculative-tool-calling/)
- Speculative Actions' OS-tuning variant is explicitly lossy (actor decision overwrites speculation, last-write-wins). — [arXiv 2510.04371](https://arxiv.org/html/2510.04371v1)

### Inferences
- Recommended student design: tiered matching. Tier 0: exact after canonicalization (sort keys, strip whitespace, normalize URLs/paths, lowercase where the tool is case-insensitive, default values filled). Tier 1 (lossy, opt-in per tool): per-argument semantic equivalence for free-text args only (embedding or normalized token set), while identifier-like args (URL, path, id) stay exact. Report accuracy impact separately for tier 1.
- Near-miss handling options: (a) discard; (b) superset reuse if the speculated call is a strict generalization (e.g. read of whole file when actor wants a slice) and slicing is deterministic; (c) keep the result in a cache keyed by canonical args for later turns. These are my suggestions, not tested in sources.

### Gaps
- No rigorous study of accuracy loss from semantic-match reuse beyond the single blog experiment.

## 6. Published next-tool-prediction numbers (summary table of cited figures)

### Takeaway
No standard benchmark exists for top-k next-tool prediction. Reported numbers come from each system's own setup; tool-name accuracy is high for self-prediction (74.6-99.6%) but full-call exact match is 20-66%.

### Cited Findings
- Speculative Actions (GPT-5-family/Gemini speculators): chess top-3 54.7%; tau-bench retail 22-38%; HotpotQA top-3 up to 46%. — [arXiv 2510.04371](https://arxiv.org/html/2510.04371v1)
- PASTE (pattern mining): top-1 up to 27.8%, top-3 recall up to 43.9%. — [arXiv 2603.18897](https://arxiv.org/html/2603.18897)
- SPORK (Qwen3-32B self-fork, name only): 74.6-99.6%; args exact 7.6-83.9%. — [arXiv 2607.03333](https://arxiv.org/html/2607.03333)
- Joint RL (Qwen3-4B, full-call exact Hit@1): 61.2 average (HotpotQA 55.2, MuSiQue 58.4, BCP 52.1, tau-Airline 68.0, tau-Retail 72.3); external drafter best 18.6. — [arXiv 2607.25816](https://arxiv.org/html/2607.25816)
- Fine-tuned Qwen3-4B (blog, tiny data): 1.9% to 51.9% name accuracy (4 tools), 31.5% (17 tools). — [yogesh.bearblog.dev](https://yogesh.bearblog.dev/speculative-tool-calling/)
- Benchmarks that relate: TRAJECT-Bench (tool-sequence exact match, accuracy drops as chains lengthen; seen only via secondary summary), T-Eval (next-step tool-call match). — [Agent evaluation survey, arXiv 2507.21504](https://arxiv.org/pdf/2507.21504), [Emergent Mind TRAJECT-Bench](https://www.emergentmind.com/topics/traject-bench)

### Inferences
- Numbers are not comparable (different models, tasks, match rules, top-k). A reasonable prior for a student project: 20-45% full-call top-3 on open-ended tasks, 50-70% on templated or structured workflows, with ~3x gap between name and full-call accuracy.

### Gaps
- No unified comparison of n-gram vs LLM predictors on one dataset (2607.25816 explicitly discusses n-gram/cached-trace methods only in related work, not as baselines).

## 7. Pitfalls and failure modes

### Takeaway
Speculation can be slower than baseline when hit rate is low, tools are fast, the actor's idle time is short, or speculative output pollutes context. Side effects are the main safety constraint.

### Cited Findings
- Naive speculation increased latency and reduced accuracy on GAIA; ODR slowed because tool-idle time <5% and draft broadened queries; empty-response loops, retry loops, premature termination, and crashes from running two models observed. — [yogesh.bearblog.dev](https://yogesh.bearblog.dev/speculative-tool-calling/)
- Wrong guesses can pollute the actor's context and derail reasoning if speculative results are injected. — [yogesh.bearblog.dev](https://yogesh.bearblog.dev/speculative-tool-calling/)
- Irreversible actions (deleting records, placing orders) unsafe without rollback/compensation; strict match penalizes equivalent calls; chess speedups vary across runs. — [arXiv 2510.04371](https://arxiv.org/html/2510.04371v1)
- Phase imbalance: blind speculation raised LLM queueing (Tool-Only 65 s queueing, 419 s E2E vs 342 s LLM-only). — [arXiv 2603.18897](https://arxiv.org/html/2603.18897)
- SPORK limits: vanishing overlap window in no-think mode (0.79x), very fast tools, format divergence (4.5% name accuracy with XML tool format on one model), probe timeouts (3.4% of BrowseComp turns), heavy batching, closed APIs. — [arXiv 2607.03333](https://arxiv.org/html/2607.03333)
- Client-side speculation needs a speculator faster than the actor with nonzero acceptance; short tools may not justify overhead; stateless cheap tools only. — [arXiv 2512.15834](https://arxiv.org/html/2512.15834)
- Speedup upper bounds: client-side below 2x; Speculative Actions at most 50% runtime reduction. — [arXiv 2512.15834](https://arxiv.org/html/2512.15834), [arXiv 2510.04371](https://arxiv.org/html/2510.04371v1)
- Distribution shift: cross-domain transfer of learned speculators held Hit@1 but task success was unstable. — [arXiv 2607.25816](https://arxiv.org/html/2607.25816)
- Fine-grained streaming can emit invalid or truncated JSON (e.g. max_tokens mid-parameter). — [Anthropic docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/fine-grained-tool-streaming)

### Inferences
- Evaluate with a replay harness first: log (context, actual call, tool latency, actor think time) from real runs, simulate each speculator offline, and compute net time saved and wasted calls before touching live agents.
- Measure the overlap window: speculation only helps if min(actor generation time remaining, tool latency) is large; compute upper bound saving = sum over turns of min(t_tool, t_actor_remaining) * P(hit).
- Guardrails: tool allowlist (idempotent/read-only), per-session speculative budget, rate limit and cost cap, never feed unverified speculative results to the actor, include cache invalidation (TTL) for tools whose results change.

### Gaps
- No incident-style reports from production teams about prefetching for agents; sources are research papers and a few blogs. Anthropic/Google engineering posts on agent prefetching were not found (Anthropic docs cover fine-grained streaming only).
