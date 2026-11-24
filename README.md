# Speculative Tool Execution for LLM Agents

**CSE 289: AI for Engineers · Team The Mockingducks**

Momin Hassan · Khizar Juma 

> AI agents spend a lot of time waiting on tools. What if we guessed the next tool call, started it early, and kept the result only if the guess was right?

---

## The idea

An LLM agent works in a loop: ask the model what to do, run a tool, wait for the result, ask again. While a tool runs, the agent sits idle, so a multi-step task can take 30 seconds or more.

Often the next step is predictable. If a search returns five links, the agent will probably open the first one. We add a small, cheap **speculator** that watches each step and guesses the next call. If the guess is confident and the tool is **read-only**, we start it early in the background and cache the result. When the agent makes the real call:

- **Match:** the cached result is returned instantly.
- **Miss:** the cached result is thrown away and the real call runs as normal.

The main agent never sees our guesses, so its reasoning is unchanged. Only its waiting time shrinks.

## How it works

```
User task
   |
   v
[Agent LLM] --- tool call ---> [Runtime] --- cache hit? --- yes ---> saved result
   ^                              |  no
   |                              v
   |                        [Real tool call]
   |                              |
   +------ observation -----------+
                |
                v   (in parallel, after every observation)
        [Speculator: small LLM or rules]
                |
        read-only AND confident?
          yes -> start tool in background -> cache
          no  -> do nothing
                |
                v
        [Log: guess, hit or miss, seconds saved]
```

## Safety rules

1. **Only read-only tools run early.** Booking and sending are never speculated.
2. **Exact match only.** A cached result is used only if the tool name and arguments match the real call exactly. Otherwise it is discarded.
3. **Low-confidence guesses are held back.**

## Repo layout

| File | What it does |
|---|---|
| `mock_tools.py` | About 10 fake tools with simulated API latency, each marked read-only or side-effecting |
| `llm_client.py` | Thin wrapper around the Gemini SDK: tool declarations, function-call parsing, 429 retry |
| `agent.py` | The agent loop, with a real Gemini brain and an offline scripted brain |
| `speculator.py` | The guessers: a rule-based one and an LLM-based one (Gemini) |
| `runtime.py` | Runs guesses early, caches results, checks matches, logs hits and misses |
| `demo.py` | Runs one task with and without speculation and compares timings |
| `tasks.py` | Train tasks (14): prompt, scripted plan, fact-based answer check, optional world-state check |
| `tasks_heldout.py` | Held-out tasks (12), shaped differently from train. Don't read while building a speculator |
| `run_eval.py` | Runs every task with and without speculation, writes a CSV to `workspace/results/` and prints a summary |
| `spec_demo.py` | Shows the speculator alone running on a real input with a real model |

## Quickstart

The real-model paths use the Google Gemini API through the official Google Gen AI SDK (`google-genai`).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Offline demo (no API key needed): scripted agent + rule-based guesser
python demo.py

# Evaluation harness (offline, fast): train tasks x 3 reps x {baseline, rules, oracle, ...}
python run_eval.py --reps 3
python run_eval.py --reps 3 --split heldout   # only once a speculator is frozen

# Live runs through the same harness (need GEMINI_API_KEY in .env; cost API calls)
python run_eval.py --modes baseline,llm,llm+prov --reps 10                 # Gemini as the guesser, scripted agent
python run_eval.py --brain gemini --scale 1 --modes baseline,llm --reps 3  # Gemini agent + Gemini guesser

# Real models: copy the env template and put your key in .env (gitignored)
cp .env.example .env        # then edit GEMINI_API_KEY=... (and optionally GEMINI_MODEL)

python spec_demo.py                         # the speculator alone on a real input
python demo.py --llm                        # Gemini agent + Gemini speculator, baseline vs speculative
python demo.py --llm --speculator rules     # Gemini agent + rule-based speculator
python demo.py --llm --log-dir logs/        # also save raw model I/O and timings as JSON
```

| Variable | Meaning |
|---|---|
| `GEMINI_API_KEY` | Required for real-model runs |
| `GEMINI_MODEL` | Model for the agent (default `gemini-pro-latest`: biggest and slowest; needs billing) |
| `GEMINI_SPEC_MODEL` | Model for the speculator (default `gemini-3.1-flash-lite`: smallest and fastest) |

All model calls go through `llm_client.py`. HTTP 429 (free-tier rate limit) and transient 5xx errors are retried with exponential backoff, and the time spent in backoff is reported by `demo.py`.

Never commit API keys. Keep them in a `.env` file, which is listed in `.gitignore`.

## Evaluation plan

26 multi-step tasks (14 train, 12 held-out), each run several times with and without speculation. Speculators are built on train tasks and scored on held-out tasks. We measure:

- total time
- how often guesses are right
- how many calls are wasted
- whether the agent's final answer stays correct

| Test group | What we expect |
|---|---|
| Search, then open a result | Many correct guesses, real time savings |
| Vague, open-ended tasks | The system mostly holds back instead of guessing |
| Tasks with a booking or send step | Side-effect tools never fire early |
| Small LLM vs. simple rules | We report honestly which is better per dollar |

The end result is a chart of **time saved vs. calls wasted**.

## Status

- [x] Mock tools, baseline agent, and speculator prototype
- [x] Runtime with cache and safety gate
- [x] 26 test tasks with a train / held-out split (plan revised from ~50 to 25-30 with a held-out split)
- [ ] Guessing and early execution working end to end *(milestone: Nov 15)*
- [ ] Full evaluation, results chart, and demo *(milestone: Dec 1)*

## Results

*Coming soon.*

## Related work

- Ye et al., [Speculative Actions: A Lossless Framework for Faster Agentic Systems](https://arxiv.org/abs/2510.04371) (ICLR 2026)
- Sui et al., [Act While Thinking: Pattern-Aware Speculative Tool Execution (PASTE)](https://arxiv.org/abs/2603.18897)
- Kim et al., [An LLM Compiler for Parallel Function Calling](https://arxiv.org/abs/2312.04511) (ICML 2024)
