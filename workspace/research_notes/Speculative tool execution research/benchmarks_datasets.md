# Benchmarks and trajectory datasets for offline evaluation of speculative tool execution

Scope note: about 19 tool calls were used. Items marked UNVERIFIED were not checked this session. Sources are repo and Hugging Face pages fetched on 2026-11-10 unless stated.

## 1. Which candidates ship downloadable recorded trajectories (tool name, args, results, order)?

### Takeaway
Good replay material exists: tau-bench historical trajectories, Nebius SWE-agent trajectories (80k), SWE-smith trajectories (76k rows), APIGen-MT-5k (5k tau-style dialogues), ToolBench (126k instances with real API results), WebArena and VisualWebArena traces, Mind2Web human traces. Several others (TheAgentCompany, MCP-Universe, BFCL, ToolSandbox) mainly produce trajectories only when you run them.

### Cited Findings
- tau-bench (sierra-research/tau-bench): MIT license. Domains are airline and retail. The folder `historical_trajectories` contains `gpt-4o-airline.json`, `gpt-4o-retail.json`, `sonnet-35-new-airline.json`, `sonnet-35-new-retail.json`. The README warns the tasks are not updated and points to tau3-bench. Install is `pip install -e .`. The default user simulator is gpt-4o and costs money to run. — [tau-bench repo](https://github.com/sierra-research/tau-bench), [folder](https://github.com/sierra-research/tau-bench/tree/main/historical_trajectories)
- tau2-bench (sierra-research/tau2-bench): the domains listed are mock, airline, retail, telecom, banking_knowledge, plus an intake domain. Setup is `uv sync` with Python 3.12 to below 3.14, using LiteLLM so any provider works. Results are saved to `data/simulations/`, and `tau2 view` browses them. The README describes no separate trajectory download. The live leaderboard is at taubench.com. The text-v1 split has 50 frozen tasks per domain (airline, retail, telecom), and voice-v1 has 50 airline, 114 retail and 114 telecom tasks. — [README](https://raw.githubusercontent.com/sierra-research/tau2-bench/main/README.md)
- tau2-bench license conflicts: one directory listing says MIT, another says the README states no license. The README fetch did not state one. Check the LICENSE file. — [ecosyste.ms](https://awesome.ecosyste.ms/projects/github.com%2Fsierra-research%2Ftau2-bench), [SourcePulse](https://www.sourcepulse.org/projects/13788360). Also, tau3-bench fixes 75+ issues, and the old and new versions cannot be installed in the same environment — [search summary](https://github.com/sierra-research/tau2-bench)
- Salesforce/APIGen-MT-5k (Hugging Face): 5,000 multi-turn trajectories, about 128 MB, CC BY-NC 4.0. ShareGPT-style JSON with turns labelled human, function_call, observation and gpt, a system prompt holding the policy, and tool descriptions. The domains are tau-bench retail and airline. Part of the data was generated with GPT-4. — [HF](https://huggingface.co/datasets/Salesforce/APIGen-MT-5k)
- nebius/SWE-agent-trajectories (Hugging Face): 80,036 trajectories, 1.11 GB, CC-BY-4.0, parquet. Columns are `instance_id`, `model_name`, `target`, `trajectory` (JSON list of messages), `exit_status`, `generated_patch`, `eval_logs`. Generators were various models, for example swe-agent-llama-70b. — [HF](https://huggingface.co/datasets/nebius/SWE-agent-trajectories)
- SWE-bench/SWE-smith-trajectories (Hugging Face): 76,002 rows across the splits tool, xml and ticks, 4.22 GB, MIT, parquet. It contains the 5,017 trajectories used to fine-tune Qwen 2.5 Coder. — [HF](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories)
- SWE-bench/experiments: the repo holds entries (metadata.yaml, results). Per-submission `trajs/` (one trace per instance, free format: md, json or yaml, no strict schema) and logs live in the submitter's own repo or an S3 bucket. `python -m analysis.download_logs` fetches them. No license is stated on the page. Submissions since 2025-11-18 need a report link. — [repo](https://github.com/SWE-bench/experiments)
- ToolBench (OpenBMB): 3,451 tools, 16,464 APIs, 126,486 instances, 469,585 real API calls, Apache-2.0 (research and education only). Data is on Google Drive and Tsinghua Cloud. Annotations include reasoning, tool execution and results. The page gives no full schema. — [repo](https://github.com/OpenBMB/ToolBench)
- StableToolBench: Apache-2.0. A cache of API responses (`tool_response_cache`, from HF `stabletoolbench/Cache` or Tsinghua Cloud) plus MirrorAPI (a model that mimics more than 7k APIs) gives deterministic replay. Six test subsets: I1 instruction, I1 category, I1 tool, I2 category, I2 instruction, I3 instruction. Cache size and task counts were not stated. — [repo](https://github.com/THUNLP-MT/StableToolBench)
- Mind2Web (osunlp): CC BY 4.0, 1,009 train rows and 6.74 GB, parquet. The test splits are Cross Task (252), Cross Website (177) and Cross Domain (about 912, shown as "9,12" on the page), and the test set is withheld and must be requested. Each action holds the pre-action HTML, the operation (CLICK, TYPE or SELECT) and candidates. Action sequences run 2 to 37 steps. — [HF](https://huggingface.co/datasets/osunlp/Mind2Web)
- WebArena: execution traces (accessibility tree, raw and parsed action, screenshots) for v1 (Aug 2023) and v2 (Nov 2023), plus human recordings for about 170 tasks, are on the project resources page. VisualWebArena GPT-4V+SoM traces cover all 910 tasks as .html files. Third-party: HF `Agent-Eval-Refine/Agent-Trajectories` (GPT-4 CoT WebArena trajectories), HF org `webarenapro` (`agent_trajectory_reviews`, contents unknown). — [search results, mirror of resources page](https://openi.pcl.ac.cn/lsyzz/web/src/branch/main/resources/README.md), [HF](https://huggingface.co/datasets/Agent-Eval-Refine/Agent-Trajectories)
- BFCL (ShishirPatil/gorilla, berkeley-function-call-leaderboard): `pip install bfcl-eval` (not the unrelated `bfcl`), Apache-2.0. Has multi-turn categories. Model outputs go in `result/MODEL_NAME/BFCL_v3_TEST_CATEGORY_result.json`. A web-search category uses SerpAPI. The page did not give the multi-turn entry count or describe the backends. — [repo](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
- ToolSandbox (apple/ToolSandbox): stateful Python tools over shared world state (settings, contacts, messages, reminders), run on the host without a sandbox. Search tools need a RapidAPI key. Output is `data/.../trajectories/conversation.json`. Install is `pip install '.[dev]'` with Python 3.9. Scenario count and license were not stated. — [repo](https://github.com/apple/ToolSandbox)
- TheAgentCompany: MIT, 175 task images. Needs Docker with 30+ GB disk and runs GitLab, Plane, ownCloud and RocketChat. Baselines ran on EC2 t3.2xlarge. The page does not say whether trajectories are released. — [repo](https://github.com/TheAgentCompany/TheAgentCompany)
- MCP-Universe (SalesforceAIResearch/MCP-Universe, PyPI `mcpuniverse`): reported as 6 domains, 11 real MCP servers, 231 tasks (133 tools is from one secondary source only). Some evaluators check live data. Best reported success is about 43.7% (GPT-5). These figures come from secondary write-ups. The first URL I tried (Salesforce-Research-ai/MCP-Universe) returned 404; the org name is SalesforceAIResearch. — [arXiv 2508.14704](https://arxiv.org/pdf/2508.14704v1), [search summary](https://socket.dev/pypi/package/mcpuniverse)

### Inferences
- Datasets with results stored inline (tau-bench, APIGen-MT, Nebius, SWE-smith, ToolBench, Mind2Web) allow replay at zero tool-execution cost. Those that need Docker or live services (TheAgentCompany, MCP-Universe, WebArena live) do not.
- The tau-bench historical trajectories come from only two models (gpt-4o and Claude 3.5 Sonnet new) and are small. Predictor tests on them may overfit to model style. APIGen-MT-5k adds volume but is NC-licensed and GPT-4-generated.

### Gaps
- Exact task counts for original tau-bench (commonly cited as 115 retail and 50 airline) were not on the fetched page.
- Number of trajectories inside each tau-bench historical JSON file, and the file sizes, were not checked.
- BFCL multi-turn entry count and whether ground-truth tool call sequences and backend state are in the repo.
- Not examined at all: GAIA, BrowseComp, AgentBench, API-Bank, HotpotQA/ReAct, OpenHands trajectories (e.g. HF all-hands datasets), Nemotron SWE trajectories (HF fetch returned 401, so existence unverified).
- Mind2Web 2 / newer web-agent trajectory sets not searched.

## 2. Read-only versus side-effect calls, and tool-call counts per task

### Takeaway
The read-only share is the key variable for speculation and is high in SWE-style and search tasks, mixed in tau-bench (lookups precede writes), and low-risk to measure in all replay settings. I could not obtain measured fractions from sources, so you must compute them from the data.

### Cited Findings
- tau-bench and tau2 domains define tools and policy per domain; the fetched pages do not classify tools as read vs write. — [tau-bench](https://github.com/sierra-research/tau-bench), [tau2](https://raw.githubusercontent.com/sierra-research/tau2-bench/main/README.md)
- ToolSandbox tools change shared world state and have implicit dependencies between tools. — [repo](https://github.com/apple/ToolSandbox)
- Mind2Web actions are CLICK, TYPE and SELECT, with 2 to 37 steps per task. — [HF](https://huggingface.co/datasets/osunlp/Mind2Web)

### Inferences
- From my background knowledge (UNVERIFIED this session): tau-bench retail and airline have get_* and list_* lookup tools (read-only) and state-changing tools such as cancel, modify, exchange, return and book. Classify by name prefix and verify against `tools/` in the repo.
- In SWE trajectories, read-only calls (open, search, ls, cat) vs edits and test runs can be separated by command, and test runs are expensive but side-effect free.

### Gaps
- No measured read-only fractions or per-task call-count statistics were found for any benchmark.

## 3. Which allow replay-based offline study (speculator on each prefix, exact vs relaxed match, zero execution cost)?

### Takeaway
Best fits: tau-bench historical trajectories plus APIGen-MT-5k (structured function calls with args), Nebius or SWE-smith trajectories (shell/edit actions, large scale), and ToolBench with StableToolBench cache (large API call sets). Mind2Web gives web actions with stored pages.

### Cited Findings
- APIGen-MT-5k has explicit `function_call` and `observation` turns, which gives the next-call label for each prefix directly. — [HF](https://huggingface.co/datasets/Salesforce/APIGen-MT-5k)
- Nebius trajectories are stored as JSON message lists with `exit_status` and a pass/fail `target`, so you can slice by success. — [HF](https://huggingface.co/datasets/nebius/SWE-agent-trajectories)
- StableToolBench's response cache plus MirrorAPI makes API results reproducible, which would let you also check speculative calls that deviate from the recording (cache hit or simulated). — [repo](https://github.com/THUNLP-MT/StableToolBench)
- Mind2Web stores pre-action HTML and candidate elements per step, but the card does not address replay. — [HF](https://huggingface.co/datasets/osunlp/Mind2Web)

### Inferences
- Pure replay only scores whether the predicted next call equals the recorded one. It cannot say what a wrong speculative call would have returned, so end-to-end speedup needs latency modelling, not real execution. For dissimilar-but-valid calls, tau-bench's deterministic in-memory DB can be re-run locally for free by replaying calls against its environment (UNVERIFIED but likely, since tau-bench tools are Python over JSON data).
- Relaxed match options: same tool name, same tool and key args, same read-only equivalence class.
- Caveat: recorded trajectories were produced by agents that were not speculation-aware, and the recorded next call is conditioned on real results.

### Gaps
- Whether tau2-bench simulations store timestamps or per-call durations was not checked.

## 4. Realistic tool latencies versus live APIs and costly infrastructure

### Takeaway
Few of these have realistic latency data. Latency will mostly have to be measured separately or modelled. Costly live setups: tau-bench and tau2 (LLM user simulator), TheAgentCompany (Docker plus EC2-class machine), MCP-Universe (live MCP servers), ToolSandbox search (RapidAPI), BFCL web search (SerpAPI).

### Cited Findings
- tau-bench default user simulator is gpt-4o, and the repo says running the benchmark can be expensive. — [tau-bench](https://github.com/sierra-research/tau-bench)
- tau2 needs a user-simulator LLM and optional judge models. — [tau2](https://raw.githubusercontent.com/sierra-research/tau2-bench/main/README.md)
- TheAgentCompany requires 30+ GB disk, Docker and several services. — [repo](https://github.com/TheAgentCompany/TheAgentCompany)
- MCP-Universe uses real MCP servers and some live-data evaluators. — [search summary](https://socket.dev/pypi/package/mcpuniverse)
- ToolBench's original data used 469,585 real API calls, but the original RapidAPI endpoints are unstable, which is why StableToolBench built a cache — [ToolBench](https://github.com/OpenBMB/ToolBench), [StableToolBench](https://github.com/THUNLP-MT/StableToolBench)

### Inferences
- A cheap approach: assign latency by tool class (e.g., DB lookup about 50 ms, web fetch about 1 s, test run about 10 s or more) and run a simulated timeline, reporting sensitivity. For SWE-style data, actual command wall time may be present in some logs (UNVERIFIED).

### Gaps
- No source gave measured tool latency for any benchmark.

## 5. Ranked shortlist and first steps

### Takeaway
1) tau-bench (plus tau2 and APIGen-MT-5k) for structured API/DB tool calls with a read/write mix. 2) Nebius SWE-agent trajectories (and SWE-smith) for long, read-heavy coding traces at scale. 3) ToolBench with StableToolBench for breadth of tools and deterministic cache. 4) Mind2Web or WebArena traces as an optional web-action case.

### Cited Findings
Concrete first steps:
1. tau-bench: `git clone https://github.com/sierra-research/tau-bench && pip install -e .`. Load `historical_trajectories/sonnet-35-new-retail.json` and the airline equivalents, extract assistant tool calls per task, build prefixes. Then pull `Salesforce/APIGen-MT-5k` via `datasets.load_dataset("Salesforce/APIGen-MT-5k")` for volume (research use only because of CC BY-NC). For live runs use tau2-bench (`uv sync`, LiteLLM, so Claude or Gemini works as agent and user simulator). — [tau-bench](https://github.com/sierra-research/tau-bench), [APIGen-MT-5k](https://huggingface.co/datasets/Salesforce/APIGen-MT-5k), [tau2](https://raw.githubusercontent.com/sierra-research/tau2-bench/main/README.md)
2. SWE: `load_dataset("nebius/SWE-agent-trajectories")` (1.11 GB, CC-BY-4.0). Filter `target == True` for successful runs, parse `trajectory` messages into actions, label read-only (find, grep, open, cat, ls) vs mutating (edit, create, test). Optional: `SWE-bench/SWE-smith-trajectories` (4.22 GB, MIT). — [Nebius](https://huggingface.co/datasets/nebius/SWE-agent-trajectories), [SWE-smith](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories)
3. ToolBench: download data from the repo's Google Drive or Tsinghua links, and the cache from HF `stabletoolbench/Cache`. Apache-2.0. — [ToolBench](https://github.com/OpenBMB/ToolBench), [StableToolBench](https://github.com/THUNLP-MT/StableToolBench)
4. Mind2Web: `load_dataset("osunlp/Mind2Web")` (CC BY 4.0). — [HF](https://huggingface.co/datasets/osunlp/Mind2Web)

### Inferences
- Skip for now: TheAgentCompany, MCP-Universe, ToolSandbox, BFCL multi-turn as primary sources (infra cost or no downloadable trajectories). BFCL and MCP-Universe could be a later live check, since BFCL is pip-installable and MCP-Universe reflects realistic MCP tools.
- Verify licenses before publishing derived data: APIGen-MT is non-commercial, Nebius carries Llama 3.1 license notes.

### Gaps
- GAIA, BrowseComp, AgentBench, API-Bank, HotpotQA/ReAct and OpenHands trajectories were not researched, so they are not ranked. From background knowledge only (UNVERIFIED): GAIA (about 466 questions) is gated on HF, BrowseComp (1,266 questions) needs live web search, and ReAct-style HotpotQA gives only short 2 to 4 call search traces.
