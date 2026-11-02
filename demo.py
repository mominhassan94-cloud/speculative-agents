"""Run one task twice (baseline vs speculative) and print timings and the speculation log.

  python demo.py                       # offline: scripted agent + rule speculator (no API key)
  python demo.py --llm                 # real Gemini agent + Gemini speculator (needs GEMINI_API_KEY)
  python demo.py --llm --speculator rules
  python demo.py --llm --log-dir reports/status_update/logs   # also save raw model I/O as JSON
"""
import argparse
import json
import os
import time
from mock_tools import TOOLS
from agent import ScriptedBrain, GeminiBrain, run_agent
from runtime import Runtime
from speculator import RuleSpeculator, LLMSpeculator, SPEC_PROMPT

TASK = ("Find the nonstop flights from Houston (IAH) to Denver (DEN) on 2026-11-06 "
        "and compare their total cost including checked-bag fees.")

p = argparse.ArgumentParser()
p.add_argument("--llm", action="store_true", help="use real Gemini for agent and speculator")
p.add_argument("--speculator", choices=["rules", "llm"], default=None)
p.add_argument("--task", default=TASK)
p.add_argument("--log-dir", default=None, help="write raw model I/O and timings as JSON here")
a = p.parse_args()

mk_brain = (lambda: GeminiBrain(TOOLS)) if a.llm else (lambda: ScriptedBrain())
spec_kind = a.speculator or ("llm" if a.llm else "rules")
mk_spec = (lambda: LLMSpeculator(TOOLS)) if spec_kind == "llm" else (lambda: RuleSpeculator())

print("TASK:", a.task, "\n")
print("=== BASELINE (no speculation) ===")
b0 = mk_brain()
ans0, t0, h0 = run_agent(a.task, b0, Runtime(TOOLS, speculator=None))
print(f"  answer: {ans0}\n  total: {t0:.1f}s\n")

print(f"=== SPECULATIVE ({spec_kind} speculator) ===")
spec = mk_spec()
b1 = mk_brain()
rt = Runtime(TOOLS, speculator=spec)
ans1, t1, h1 = run_agent(a.task, b1, rt)
print(f"  answer: {ans1}\n  total: {t1:.1f}s\n")
time.sleep(0.5)  # let any in-flight speculation finish logging

calls0 = [(h["tool"], h["args"]) for h in h0]
calls1 = [(h["tool"], h["args"]) for h in h1]
print("=== SPECULATION LOG ===")
for line in rt.log:
    print(" ", line)
print("\n=== SUMMARY ===")
print(" ", rt.summary())
print(f"  time: {t0:.1f}s -> {t1:.1f}s  ({(1 - t1 / t0) * 100:.0f}% faster)")
print("  same tool-call sequence as baseline:", calls0 == calls1)
print("  same final answer as baseline:", ans0 == ans1)
if a.llm:
    from llm_client import RETRIES
    print(f"  rate-limit retries: {RETRIES.retries} ({RETRIES.sleep_s:.1f}s spent in backoff, included in times above)")

if a.log_dir:
    os.makedirs(a.log_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = {
        "task": a.task, "agent": getattr(b0, "model", b0.name), "speculator": spec_kind,
        "speculator_model": getattr(spec, "model", None),
        "speculator_system_prompt": SPEC_PROMPT if spec_kind == "llm" else None,
        "baseline": {"total_s": round(t0, 2), "answer": ans0, "calls": calls0,
                     "model_turns": getattr(b0, "raw_log", None)},
        "speculative": {"total_s": round(t1, 2), "answer": ans1, "calls": calls1,
                        "model_turns": getattr(b1, "raw_log", None),
                        "speculator_calls": getattr(spec, "calls", None),
                        "speculation_log": rt.log, "summary": rt.summary()},
    }
    if a.llm:
        from llm_client import RETRIES
        out["rate_limit_retries"] = {"count": RETRIES.retries, "backoff_s": round(RETRIES.sleep_s, 2)}
    path = os.path.join(a.log_dir, f"demo-{spec_kind}-{stamp}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("  log written to", path)
