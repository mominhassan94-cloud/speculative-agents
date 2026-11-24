"""Thin wrapper around the Google Gen AI SDK so the rest of the code never touches SDK details.

Config comes from the environment (or a .env file in the repo root, which is gitignored):
  GEMINI_API_KEY     required for any real-model run
  GEMINI_MODEL       model for the agent and the speculator (default below)
  GEMINI_SPEC_MODEL  optional: a different model for the speculator only
"""
import json
import os
import random
import threading
import time

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

DEFAULT_MODEL = "gemini-3.5-flash-lite"
RETRY_CODES = {429, 500, 503}   # 429 = free-tier rate limit; 500/503 = transient server errors


def agent_model():
    return os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)


def spec_model():
    return os.environ.get("GEMINI_SPEC_MODEL") or agent_model()


class RetryStats:
    """Counts rate-limit retries so timing comparisons can report how much time went to backoff."""

    def __init__(self):
        self.lock = threading.Lock()
        self.retries = 0
        self.sleep_s = 0.0

    def add(self, s):
        with self.lock:
            self.retries += 1
            self.sleep_s += s


RETRIES = RetryStats()


class LLMClient:
    def __init__(self, model=None, max_attempts=6):
        from google import genai
        from google.genai import types
        self.types = types
        self.client = genai.Client()  # reads GEMINI_API_KEY from the environment
        self.model = model or agent_model()
        self.max_attempts = max_attempts

    def _generate(self, contents, config):
        from google.genai import errors
        delay = 2.0
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self.client.models.generate_content(model=self.model, contents=contents, config=config)
            except errors.APIError as e:
                if e.code not in RETRY_CODES or attempt == self.max_attempts:
                    raise
                wait = delay + random.uniform(0, 1)
                print(f"  [llm_client] HTTP {e.code} from {self.model}, retry {attempt} in {wait:.1f}s")
                RETRIES.add(wait)
                time.sleep(wait)
                delay = min(delay * 2, 60)

    # --- plain text completion (used by the speculator) -------------------------------
    def complete(self, system, user, max_tokens=300):
        t = self.types
        cfg = t.GenerateContentConfig(
            system_instruction=system, max_output_tokens=max_tokens,
            thinking_config=t.ThinkingConfig(thinking_level=t.ThinkingLevel.MINIMAL))
        return self._generate(user, cfg).text or ""

    # --- tool calling (used by the agent) --------------------------------------------
    def tool_config(self, tools, max_tokens=1024):
        """Our Tool registry (Anthropic-style input_schema) -> Gemini function declarations."""
        t = self.types
        decls = [t.FunctionDeclaration(name=x.name, description=x.description,
                                       parameters_json_schema=x.input_schema) for x in tools.values()]
        return t.GenerateContentConfig(
            tools=[t.Tool(function_declarations=decls)],
            automatic_function_calling=t.AutomaticFunctionCallingConfig(disable=True),  # our runtime runs tools
            max_output_tokens=max_tokens,
            thinking_config=t.ThinkingConfig(thinking_level=t.ThinkingLevel.LOW))

    def user_turn(self, text):
        return self.types.Content(role="user", parts=[self.types.Part(text=text)])

    def tool_result_turn(self, name, result):
        """Equivalent of an Anthropic tool_result block: a user turn holding a function response."""
        return self.types.Content(role="user", parts=[
            self.types.Part.from_function_response(name=name, response={"result": result})])

    def step(self, contents, config):
        """One agent turn. Returns (model_content, action, raw).

        model_content must be appended to the history unchanged: Gemini 3 models attach thought
        signatures to function-call parts and expect them back on the next turn.
        action is {"tool", "args"} or {"final"}; raw is a JSON-safe dump of the reply for logging.
        """
        r = self._generate(contents, config)
        content = r.candidates[0].content if r.candidates else None
        parts = (content.parts or []) if content else []
        raw = {"finish_reason": str(r.candidates[0].finish_reason) if r.candidates else None,
               "parts": [_part_json(p) for p in parts]}
        for i, p in enumerate(parts):
            if p.function_call:
                # Gemini may request several calls at once (parallel function calling). Our loop runs one
                # tool per step, so keep the turn only up to the first call: history then holds exactly one
                # call per tool result (as the Anthropic version did). raw still logs every requested call.
                content = self.types.Content(role=content.role, parts=parts[:i + 1])
                return content, {"tool": p.function_call.name, "args": dict(p.function_call.args or {})}, raw
        text = "".join(p.text for p in parts if p.text and not p.thought)
        return content, {"final": text}, raw


def _part_json(p):
    if p.function_call:
        return {"function_call": {"name": p.function_call.name, "args": dict(p.function_call.args or {})},
                "has_thought_signature": p.thought_signature is not None}
    if p.text is not None:
        return {"thought" if p.thought else "text": p.text}
    return {"other": json.loads(p.model_dump_json(exclude_none=True, exclude={"thought_signature"}))}
