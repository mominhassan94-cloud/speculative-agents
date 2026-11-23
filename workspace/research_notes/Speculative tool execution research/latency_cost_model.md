# Latency and cost model for speculative tool execution (as of 2026-11-10)

Labels: [M] = measurement reported by a source; [V] = vendor claim; [E] = my estimate or derivation (not sourced); [Gap] = not found.
Caveat: several sources below were read through search-result summaries or a fetch-tool summary, not the full page. Re-check any number you plan to headline.

## 1. Tool-call latency distributions and tails

### Takeaway
Web-search APIs sit at roughly 0.5-2.5 s median with p95 about 1.5-2x the median (sigma of about 0.4 on a log scale). Sandboxes can be fast (about 0.1-0.7 s) but have heavy cold-start tails. Only web search has decent public p50/p95 data. For page fetch, SQL, file reads, MCP, internal REST and vector search I found no citable agent-oriented p50/p95, so those rows are estimates. The mock range of 0.7-2.0 s is plausible for web-type tools, but a uniform distribution has too thin a tail.

### Cited Findings
- Google-SERP APIs, Feb 2026, p50 / p95, concurrency not stated: Serper 2.23 / 4.21 s; SerpApi 2.53 / 5.27 s; Scrapingdog 2.48 / 6.82 s; SearchApi 2.71 / 8.28 s; DataForSEO 4.54 / 10.73 s; Zenserp 3.92 / 11.36 s. Bright Data and SerpApi also list about 0.9 s "premium routing" medians. [M, vendor-authored comparison] — [HackerNoon SERP benchmarks, 2026-02-24](https://hackernoon.com/lite/serp-benchmarks-success-rates-and-latency-at-scale)
- Tavily: mean 2.0 s, median 1.724 s, p95 3.534 s over 100 concurrent queries. [M, run by a competitor, fastCRW] — [fastCRW Tavily benchmark](https://fastcrw.com/benchmarks/tavily-search)
- Tavily "advanced" mode about 4.2 s median and Exa median 569 ms (300-query write-up, no p95 computed). [M, secondary] — [Effloow Exa vs Tavily 2026](https://effloow.com/articles/exa-vs-tavily-agent-web-search-benchmark-2026)
- Brave about 561 ms mean (Telnyx test) and 669 ms in another comparison; Exa 1.2 s in that comparison. Test setups differ, so these are not comparable. [M, secondary] — [Telnyx web search API benchmark](https://telnyx.com/resources/web-search-api-benchmark.md); [DEV comparison](https://dev.to/supertrained/exa-vs-tavily-vs-serper-vs-brave-search-for-ai-agents-an-score-comparison-2l1g)
- Exa claims any search API wrapping Google has a minimum p50 of about 700 ms. [V, competitor claim] — [Exa Instant](https://exa.ai/blog/exa-instant)
- Sandbox start: Daytona advertises sub-90 ms creation, E2B about 150 ms, Modal sub-second. None publish p50/p99. [V] — [Fly.io sandbox comparison](https://fly.io/learn/sprites-vs-e2b-vs-modal-vs-daytona/); [StartupHub 2026](https://www.startuphub.ai/news/daytona-vs-e2b-vs-modal-vs-vercel-sandbox-2026)
- Sandbox resume (not cold start): E2B 662 ms, Modal snapshot boot 2347 ms, Blaxel 1924 ms, Vercel 3333 ms. [M] — [Superagent sandbox benchmark 2026](https://www.superagent.sh/blog/ai-code-sandbox-benchmark-2026)
- E2B cold start p50 440 ms, p99 3,150 ms (about 7x tail). [competitor marketing, treat as weak] — [Rivet agentOS performance](https://rivet.dev/agentos/docs/performance)
- Tool share of agent latency varies by tool type. In SWE-Agent, Bash/Python execution took 38% and 25% of E2E latency on APPS and BigCodeBench respectively. [M, via search summary] — [arXiv 2511.00739](https://arxiv.org/pdf/2511.00739)

### Inferences
- Across web-search providers, log-scale sigma = ln(p95/p50)/1.645: Serper 0.39, Tavily 0.44, SerpApi 0.45, SearchApi about 0.68, DataForSEO about 0.52. Typical sigma is 0.4-0.7 for web APIs. [E, computed from the cited numbers]
- Serverless sandbox start has a heavier tail (p99/p50 of about 7 in one vendor-contested datapoint). Treat sigma of 0.8-1.0 as plausible. [E]
- Rate-limit and 429 retry behaviour is not in any of these latency numbers. It would add a discrete tail.

### Estimates for rows with no citable data (label as guesses in the report)
| Tool | p50 | p95 | Basis |
|---|---|---|---|
| Local file read / grep | 1-20 ms | 50-200 ms | [E] |
| SQL, indexed OLTP query | 5-30 ms | 100-300 ms | [E] |
| Vector search, managed ANN | 20-80 ms | 150-400 ms | [E] |
| Internal REST | 50-300 ms | 0.5-1.5 s | [E] |
| MCP tool call (adds JSON-RPC and process or HTTP hop on top of the underlying tool) | underlying + 5-50 ms | underlying + 200 ms | [E] |
| Page fetch / browse (static) | 0.5-1.5 s | 3-8 s | [E] |
| Headless-browser render | 2-5 s | 10-20 s | [E] |
| Test run / build in sandbox | 5-60 s | minutes | [E] |

### Gaps
- No primary p50/p95 for page fetch, browser, SQL, vector DB, MCP overhead or internal REST. I ran one sandbox query, and the MCP and vector-DB parts of it did not get searched. Measure in the target deployment.
- Published search benchmarks are mostly vendor- or competitor-authored, and concurrency is rarely stated.

## 2. LLM step latency and tool-vs-LLM time share

### Takeaway
In published traces, tool execution is a large share of end-to-end time (about 13-65% depending on agent, benchmark and hardware), so tool time is a first-order target. Reliable per-step TTFT and tokens/s for Sonnet 5.5 and Haiku 5.5 were not obtainable. The only Artificial Analysis rows I got are reasoning-mode and clearly not usable as tool-loop latency.

### Cited Findings
- PASTE: tool execution is 45-57% of agent E2E latency with no queueing. Going from 1 to 192 concurrent sessions, LLM generation time grows more than 17x while tool time changes modestly. In one workload exposed tool time fell from 84-89 s to 34 s and LLM-side time from 226-249 s to 186 s. [M, authors' evaluation on Qwen3-30B-A3B and Qwen-DeepResearch-30B, A100 cluster] — [PASTE arXiv 2603.18897](https://arxiv.org/html/2603.18897) (fetched; summarised by a fetch model)
- A separate profiling study reports tool execution at 13-48% of total latency across datasets and harnesses, with the top end on Terminal-Bench 2. [M, via search summary] — arXiv 2607.29069 or 2608.15127 (the summary did not say which); see [arXiv 2511.00739](https://arxiv.org/pdf/2511.00739) for the SWE-Agent figures above
- CPU-centric study: on a faster GPU system, tool share rises to up to 65% of E2E latency because faster inference exposes tool time. [M, via search summary] — [arXiv 2511.00739](https://arxiv.org/pdf/2511.00739)
- Counterexample: a multi-agent workflow study with simulated tool delays had LLM plus orchestration at about 91% and tool I/O at about 9%. [M but synthetic] — arXiv 2606.19382 (via the same search summary)
- Speculative Actions paper: speculators with low reasoning budgets answer in about 2-3 s (external leaderboard); a chess actor at high reasoning effort is described only as "high latency". [weak] — [arXiv 2510.04371](https://arxiv.org/html/2510.04371v1)
- Artificial Analysis comparison page: Claude Haiku 5.5 (Max, reasoning) shows TTFT 181.5 s and 254 tok/s; Gemini 3.8 Flash (High, reasoning) shows TTFT 17.09 s and 113 tok/s. The TTFT values include reasoning time and are not comparable to a tool-call step. [M, effort labels undefined on the page] — [Artificial Analysis](https://artificialanalysis.ai/models/comparisons/claude-haiku-5-5-vs-gemini-3-8-flash)
- Older reference: Haiku 4.5 TTFT about 597 ms, Gemini 2.5 Flash about 204.5 tok/s. [M, secondary, outdated] — [DEV Community 2026](https://dev.to/kunal_d6a8fea2309e1571ee7/5-llm-apis-tested-for-latency-real-data-2026-3e4o)

### Inferences
- A tool-calling step is TTFT + output_tokens / speed. With 150-400 output tokens (reasoning plus tool JSON), a Sonnet-class step at roughly 50-100 tok/s is about 3-6 s, and a Haiku-class step at roughly 200+ tok/s is about 0.7-2 s. [E; the speeds are not sourced for Sonnet 5.5]
- Therefore the simulator's 1.5 s "think time" is closer to a Haiku-class, no-reasoning step than a Sonnet-class one. A realistic think-time distribution for a Sonnet-class actor is lognormal with median about 3-4 s and p95 about 8-10 s (add extended-thinking steps at 10-60 s). [E]

### Gaps
- No clean non-reasoning TTFT or tok/s for Sonnet 5.5, Haiku 5.5 or Gemini Flash. Open the Artificial Analysis model pages directly, or run a timed call. I did not retrieve the SWE-agent paper's own time breakdown.

## 3. Speculator overhead and token pricing

### Takeaway
A Haiku 5.5-class speculator call costs on the order of $0.0002-0.0005 per step, one to two orders of magnitude below a Sonnet 5.5 actor step, so dollar overhead is small relative to the main loop. The larger cost is wasted tool calls that bill per request, plus tokens when a speculator runs on every step.

### Cited Findings
Official Anthropic pricing page (fetched 2026-11-10), $ per MTok, base input / 5m cache write / cache read / output:
- Claude Haiku 5.5, prompts up to 100k tokens: 0.10 / 0.125 / 0.01 / 0.50. Over 100k tokens: 0.50 / 0.625 / 0.05 / 2.50. — [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing)
- Claude Haiku 4.5: 1 / 1.25 / 0.10 / 5. — same page
- Claude Sonnet 5.5: 2 / 2.50 / 0.10 / 10. Sonnet 5: 2 / 2.50 / 0.20 / 10. Sonnet 4.6: 3 / 3.75 / 0.30 / 15. — same page
- Opus 5.5: 4 / 5 / 0.20 / 20. Fable 5.2: 8 / 10 / 0.20 / 40. — same page
- Batch API gives a 50% discount on input and output; fast mode is not available with batch. — same page
- Artificial Analysis lists Gemini 3.8 Flash at $0.75 in / $3.75 out per 1M. — [Artificial Analysis](https://artificialanalysis.ai/models/comparisons/claude-haiku-5-5-vs-gemini-3-8-flash)
- Speculative Actions OS-tuning case (cumulative at 200 s): Actor-only 63k tokens / 2.18 cents; Speculator-only 2.10M tokens / 21.5 cents; Actor + Speculator 608k tokens / 7.83 cents. So speculation roughly tripled the cost in that case (about 3.6x). — [arXiv 2510.04371](https://arxiv.org/html/2510.04371v1)
- PASTE: predictor and scheduler add under 100 ms; 1-3 idle CPU cores and 250 MB extra memory bought a 48% tool-latency drop; more than 20,000 speculative actions were generated, of which 602 were flagged potentially side-effecting and blocked. — [PASTE](https://arxiv.org/html/2603.18897)

### Inferences (worked per-step costs) [E; assumes 2,000 input tokens uncached, 100 output tokens]
- Haiku 5.5: 2000 x 0.10e-6 + 100 x 0.50e-6 = $0.00025 per speculative call. With a cached prefix the input part falls further (cache read is $0.01/MTok).
- Haiku 4.5: $0.0025 per call.
- Sonnet 5.5 actor step with 8,000 input and 300 output tokens: 8000 x 2e-6 + 300 x 10e-6 = $0.019. So a Haiku 5.5 speculator is about 1.3% of an actor step.
- Speculator latency, assuming about 0.4-0.6 s TTFT and 100 tokens at about 250 tok/s: roughly 0.8-1.0 s. [E; no source for non-reasoning TTFT.] Short outputs (tool name plus args only, no reasoning) matter more than model size.

### Gaps
- Tool-use system-prompt token overhead is on the pricing page (about 300+ tokens per request) but I did not extract the exact figure.
- Web-search API per-call prices (Tavily, Serper, Brave) were not retrieved; needed for wasted-call cost.

## 4. Analytical model

### Takeaway
Per step, the saving on a correct speculation is min(T, D - s), where T is tool latency, D is the actor's think time and s is the speculator latency. Expected saving is p x that minus overheads, so speculation pays only when the speculator is faster than the actor and the tool is slow enough to matter.

### Derivation [E; consistent with the bound in Speculative Actions]
Notation: T = tool latency, D = actor (LLM) step time, s = speculator latency (launched at the start of the step, in parallel with the actor), p = probability the speculated call equals the actor's call, c_s = speculator call cost, c_t = tool cost per call, v = value of one second of latency in dollars.
1. Sequential step time = D + T.
2. On a hit, the speculated tool starts at s and ends at s + T; the step ends at max(D, s + T). Saving = (D + T) - max(D, s + T) = min(T, D - s), valid for D > s, else 0.
3. On a miss, the real call runs after the actor as usual, so there is no time loss if capacity is free. With contention penalty delta (queueing or rate-limit delay for the real call caused by the speculative one) the miss costs delta.
4. Expected time saving per step: E[S] = p x min(T, D - s) - (1 - p) x delta.
5. Expected extra dollar cost per step: E[C] = c_s + (1 - p) x c_t (wasted tool call on a miss; the hit call is not wasted, it replaces the real one).
6. Break-even: p x min(T, D - s) x v >= c_s + (1 - p) x c_t + (1 - p) x delta x v.
   - Latency-only (cost ignored, delta = 0): any p > 0 helps provided D > s. Gain is capped at T, and at D - s if the tool is slower than the actor.
   - Dollar-bound: p* = (c_s + c_t) / (min(T, D - s) x v + c_t).
7. Worked example [E]: T = 1.35 s (mock mean), D = 1.5 s, s = 0.9 s. min(T, D - s) = 0.6 s, so even a perfect predictor saves only 0.6 s of a 2.85 s step (21%). At p = 0.3 the saving is about 0.18 s (6%). With D = 4 s, s = 0.9 s the cap becomes T = 1.35 s: at p = 0.3, 0.4 s saved of 5.35 s (7.6%); at p = 0.55 (best Speculative Actions chess accuracy), 0.74 s (14%).
8. Whole-run form, all steps iid: speedup = (D + T) / (D + T - p x min(T, D - s)). Limit p = 1, tool-bound: 1 + T/D... capped by the speculator rate (Speculative Actions gives a 50% bound at p = 1).
9. Speculative Actions closed form: E[T_s]/E[T_seq] -> 1 - (p/(1+p)) x (alpha/(alpha+beta)), where alpha is the speculator rate and beta the actor rate. Their measured result: about 20% end-to-end lossless speedup, 19.5% on chess with 3 predictions. — [arXiv 2510.04371](https://arxiv.org/html/2510.04371v1)
10. Multiple candidates (top-k): hit rate rises to p_k (PASTE reports top-1 27.8% but top-3 recall 43.9% and an overall hit rate up to 93.8% with its pattern scheme), but cost rises to k x c_t on speculative calls and contention delta rises. — [PASTE](https://arxiv.org/html/2603.18897)

### Concurrency, rate limits and caching [E unless cited]
- Concurrency caps (workers, sandbox slots, API QPS) turn delta from zero into a real cost: a speculative call that occupies the last slot delays the real one. Track delta as a function of utilisation. PASTE found speculation use only idle CPU (1-3 cores) and uses a risk-aware scheduler to keep authoritative tools at full performance. — [PASTE](https://arxiv.org/html/2603.18897)
- Under high LLM concurrency (192 sessions) LLM time grew 17x while tool time barely moved, so with a shared LLM server the speculator itself competes for the same GPU and its s grows; the model above needs s measured under load. — [PASTE](https://arxiv.org/html/2603.18897)
- Rate limits: per-provider QPS and monthly quotas for Tavily, Serper and Brave were not retrieved [Gap]. Wasted calls consume quota, so cost c_t should include quota value, not only per-call price.
- Caching: if a speculated call is identical to a later or earlier real call (same key), a hit costs nothing extra and a miss is not wasted if the result is cached for later reuse. Effective waste = (1 - p) x c_t x (1 - r), where r is the probability a missed result is reused within TTL. Prefetch of idempotent reads (search, file read) is safe; side-effecting calls must not be speculated (PASTE blocked all 602 flagged cases).

### Gaps
- Real hit rates for our own tasks are unknown; the paper values (22-55% top-1 style accuracy in Speculative Actions; 27.8% top-1 and up to 93.8% overall hit in PASTE) come from different definitions and workloads and are not interchangeable.

## 5. Are the mock latencies realistic, and what to use instead

### Takeaway
The 0.7-2.0 s tool range is the right order of magnitude for web search and fetch, but wrong in shape (uniform, thin tail) and wrong for fast tools (files, SQL, vector search). The 1.5 s think time is optimistic for a Sonnet-class actor and makes the speculator look less useful than it would be (it caps the saving at D - s).

### Cited Findings
- Web-search medians are 0.6-2.5 s with p95 of 1.5-2x the median in the data above; the mock mean of 1.35 s and p95 of about 1.9 s (uniform) is in range but its p95/p50 ratio of about 1.4 is lower than the 1.9 seen for Serper and Tavily (4.21/2.23 and 3.53/1.72). — [HackerNoon](https://hackernoon.com/lite/serp-benchmarks-success-rates-and-latency-at-scale); [fastCRW](https://fastcrw.com/benchmarks/tavily-search)

### Inferences: suggested simulator distributions [E; parameters derived from the cited p50/p95 where noted]
Use lognormal with latency = median x exp(sigma x Z), Z ~ N(0,1); optionally clip at a timeout and add a small timeout/retry mass.
| Tool class | median (s) | sigma | resulting p95 (s) | basis |
|---|---|---|---|---|
| Web search (Serper-like) | 2.2 | 0.39 | 4.2 | cited (HackerNoon) |
| Web search (Tavily-like) | 1.7 | 0.44 | 3.5 | cited (fastCRW) |
| Web search (Brave/Exa-like, fast) | 0.6 | 0.45 | 1.3 | weak sources |
| Page fetch | 1.0 | 0.7 | 3.2 | guess |
| Sandbox code exec | 0.5 | 0.9 | 2.2 | guess |
| DB / vector / file | 0.03-0.05 | 0.8 | 0.12-0.19 | guess |
| Internal REST | 0.15 | 0.7 | 0.47 | guess |
| Actor think time (Sonnet-class, tool-call step) | 3.5 | 0.55 | 8.6 | guess |
| Actor think time (Haiku-class) | 1.2 | 0.5 | 2.7 | guess |
| Speculator (Haiku-class, short output) | 0.8 | 0.4 | 1.5 | guess |
- Add a 1-3% chance of a "slow tail" outlier at 5-10x the median (timeouts, retries, 429s).
- Run the sweep at both think-time settings (1.5 s and about 3.5 s median). With D = 1.5 s and s = 0.8 s the cap on saving per step is 0.7 s regardless of hit rate, so results may understate real benefit with slower actors.
- Add contention: model a concurrency cap so speculative calls can delay real ones (delta), and a per-call cost so wasted cost is reported.

### Gaps
- I did not inspect the repo's simulator code (the grep command failed on my shell glob), so the "0.7-2.0 s tool / 1.5 s think" values are taken from the task brief.

## Source reliability notes
- Pricing numbers come from Anthropic's official pricing page; model names in it (Sonnet 5.5, Haiku 5.5) were read from that page and match the date of this work.
- Search-latency benchmarks are vendor- or competitor-authored; sandbox numbers are largely vendor claims.
- Artificial Analysis figures retrieved were reasoning-mode only; do not use their TTFT as a tool-loop step latency.
- PASTE and Speculative Actions numbers are the authors' own and were read via a fetch model's summary; the Speculative Actions text itself notes an inconsistency (5 vs 55 runs) in one figure.
