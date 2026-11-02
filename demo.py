"""Run one task twice (baseline vs speculative) and print timings and the speculation log.

  python demo.py                 # offline: scripted agent + rule speculator (no API key)
  python demo.py --llm           # real Claude agent + real Claude speculator (needs ANTHROPIC_API_KEY)
"""
import argparse
from mock_tools import TOOLS
from agent import ScriptedBrain, AnthropicBrain, run_agent
from runtime import Runtime
from speculator import RuleSpeculator, LLMSpeculator

TASK = ("Find the nonstop flights from Houston (IAH) to Denver (DEN) on 2026-11-06 "
        "and compare their total cost including checked-bag fees.")

p = argparse.ArgumentParser()
p.add_argument("--llm", action="store_true", help="use real Claude for agent and speculator")
p.add_argument("--speculator", choices=["rules", "llm"], default=None)
a = p.parse_args()

mk_brain = (lambda: AnthropicBrain(TOOLS)) if a.llm else (lambda: ScriptedBrain())
spec_kind = a.speculator or ("llm" if a.llm else "rules")
mk_spec = (lambda: LLMSpeculator(TOOLS)) if spec_kind == "llm" else (lambda: RuleSpeculator())

print("TASK:", TASK, "\n")
print("=== BASELINE (no speculation) ===")
ans0, t0, _ = run_agent(TASK, mk_brain(), Runtime(TOOLS, speculator=None))
print(f"  answer: {ans0}\n  total: {t0:.1f}s\n")

print(f"=== SPECULATIVE ({spec_kind} speculator) ===")
rt = Runtime(TOOLS, speculator=mk_spec())
ans1, t1, _ = run_agent(TASK, mk_brain(), rt)
print(f"  answer: {ans1}\n  total: {t1:.1f}s\n")

print("=== SPECULATION LOG ===")
for line in rt.log:
    print(" ", line)
print("\n=== SUMMARY ===")
print(" ", rt.summary())
print(f"  time: {t0:.1f}s -> {t1:.1f}s  ({(1 - t1 / t0) * 100:.0f}% faster)")
print("  same final answer as baseline:", ans0 == ans1)
