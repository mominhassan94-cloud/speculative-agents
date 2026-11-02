"""One working piece: the guesser (speculator) running on a real input with a real Gemini model.

  python spec_demo.py        (needs GEMINI_API_KEY; model from GEMINI_SPEC_MODEL or GEMINI_MODEL)
"""
from mock_tools import TOOLS
from speculator import LLMSpeculator, SPEC_PROMPT

TASK = ("Find the nonstop flights from Houston (IAH) to Denver (DEN) on 2026-11-06 "
        "and compare their total cost including checked-bag fees.")

# Real input: the agent has just made its first tool call and got a result back.
args = {"origin": "IAH", "dest": "DEN", "date": "2026-11-06"}
history = [{"tool": "search_flights", "args": args, "result": TOOLS["search_flights"].fn(**args)}]

spec = LLMSpeculator(TOOLS)
guess = spec.predict(TASK, history)

print("=== MODEL ===\n" + spec.model)
print("\n=== PROMPT (system) ===\n" + SPEC_PROMPT)
print("\n=== INPUT (user message sent to the model) ===\n" + spec.last_user)
print("\n=== MODEL'S RAW OUTPUT ===\n" + spec.last_raw)
print("\n=== PARSED GUESS ===\n" + str(guess))
