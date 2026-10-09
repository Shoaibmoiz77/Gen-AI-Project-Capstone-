"""LLM clients.

Every model call in this project asks for a *forced tool call* rather than free text. The
tool's JSON schema is the output contract, so we get structured answers (sentences, citation
ids, abstention flags, judge scores) without fragile regex parsing of prose.

``FakeLLM`` implements the same interface deterministically, so tests and the offline eval
run with no API key and no network.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from groundwork.text import tokenize


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens, self.output_tokens + other.output_tokens
        )


@dataclass
class ToolResult:
    input: dict[str, Any]
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0


class LLM(Protocol):
    name: str

    def call_tool(self, system: str, user: str, tool: dict[str, Any]) -> ToolResult: ...


class AnthropicLLM:
    def __init__(self, model: str, max_tokens: int = 1024, client: Any | None = None) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic(max_retries=3)
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.name = model

    def call_tool(self, system: str, user: str, tool: dict[str, Any]) -> ToolResult:
        start = time.perf_counter()
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
        )
        latency = (time.perf_counter() - start) * 1000
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use" and block.name == tool["name"]:
                return ToolResult(
                    input=dict(block.input),
                    usage=Usage(resp.usage.input_tokens, resp.usage.output_tokens),
                    latency_ms=latency,
                )
        raise RuntimeError(f"Model did not call tool {tool['name']!r}")


# --------------------------------------------------------------------------------------------
# Any OpenAI-compatible chat API: Groq, Gemini, Ollama, OpenRouter, vLLM, ...
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Provider:
    base_url: str
    default_model: str
    key_env: str | None  # None = no key needed (local server)


PROVIDERS: dict[str, Provider] = {
    "groq": Provider(
        "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile", "GROQ_API_KEY"
    ),
    "gemini": Provider(
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "gemini-2.5-flash",
        "GEMINI_API_KEY",
    ),
    "ollama": Provider("http://localhost:11434/v1", "llama3.1", None),
}


class OpenAICompatLLM:
    """Forced function calling over the OpenAI chat-completions wire format.

    Free tiers rate-limit aggressively, so 429s and 5xx responses are retried with
    exponential backoff (honouring ``Retry-After`` when the server sends it).
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        max_tokens: int = 1024,
        max_retries: int = 6,
        client: Any | None = None,
    ) -> None:
        import httpx

        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = client or httpx.Client(timeout=120, headers=headers)
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.name = model

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            r = self.client.post(self.url, json=payload)
            if r.status_code == 429 or r.status_code >= 500:
                if attempt == self.max_retries:
                    break
                retry_after = r.headers.get("retry-after")
                try:
                    wait = float(retry_after) if retry_after else delay
                except ValueError:
                    wait = delay
                time.sleep(min(wait, 60))
                delay *= 2
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"{r.status_code} from {self.url}: {r.text[:500]}")
            return r.json()
        raise RuntimeError(f"Rate limited or unavailable after {self.max_retries} retries: "
                           f"{r.status_code} {r.text[:300]}")

    def call_tool(self, system: str, user: str, tool: dict[str, Any]) -> ToolResult:
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool["input_schema"],
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": tool["name"]}},
        }
        start = time.perf_counter()
        data = self._post(payload)
        latency = (time.perf_counter() - start) * 1000

        message = data["choices"][0]["message"]
        for call in message.get("tool_calls") or []:
            fn = call.get("function", {})
            if fn.get("name") == tool["name"]:
                args = fn.get("arguments") or "{}"
                parsed = json.loads(args) if isinstance(args, str) else dict(args)
                usage = data.get("usage") or {}
                return ToolResult(
                    input=parsed,
                    usage=Usage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)),
                    latency_ms=latency,
                )
        raise RuntimeError(f"Model did not call tool {tool['name']!r}")


# --------------------------------------------------------------------------------------------
# Deterministic stand-in used by tests and offline evaluation
# --------------------------------------------------------------------------------------------

SOURCE_RE = re.compile(r'<source id="(\d+)"[^>]*>\n?(.*?)\n?</source>', re.S)
QUESTION_RE = re.compile(r"<question>\n?(.*?)\n?</question>", re.S)


def _overlap(a: str, b: str) -> float:
    ta, tb = set(tokenize(a)), set(tokenize(b))
    return len(ta & tb) / len(ta) if ta else 0.0


class FakeLLM:
    """Extractive baseline that speaks the same tool protocol as the real model.

    * answer tool: returns the single sentence from the sources with the highest token overlap
      with the question, cited to its source. Abstains when overlap is below a threshold.
    * judge tool: scores faithfulness by token overlap between answer and cited text.
    """

    name = "fake-extractive"

    def __init__(self, abstain_below: float = 0.34) -> None:
        self.abstain_below = abstain_below

    def call_tool(self, system: str, user: str, tool: dict[str, Any]) -> ToolResult:
        if tool["name"] == "submit_answer":
            return ToolResult(input=self._answer(user))
        if tool["name"] == "submit_grade":
            return ToolResult(input=self._grade(user))
        raise ValueError(f"FakeLLM does not know tool {tool['name']!r}")

    def _answer(self, user: str) -> dict[str, Any]:
        q_match = QUESTION_RE.search(user)
        question = q_match.group(1) if q_match else user
        best: tuple[float, str, int] = (0.0, "", 0)
        for sid, body in SOURCE_RE.findall(user):
            for sent in re.split(r"(?<=[.!?])\s+|\n+", body):
                sent = sent.strip(" -*#")
                if len(sent) < 20:
                    continue
                score = _overlap(question, sent)
                if score > best[0]:
                    best = (score, sent, int(sid))
        if best[0] < self.abstain_below:
            return {"answerable": False, "sentences": []}
        return {"answerable": True, "sentences": [{"text": best[1], "citations": [best[2]]}]}

    def _grade(self, user: str) -> dict[str, Any]:
        answer = re.search(r"<answer>\n?(.*?)\n?</answer>", user, re.S)
        ref = re.search(r"<reference_answer>\n?(.*?)\n?</reference_answer>", user, re.S)
        ctx = re.search(r"<cited_sources>\n?(.*?)\n?</cited_sources>", user, re.S)
        a = answer.group(1) if answer else ""
        faith = _overlap(a, ctx.group(1) if ctx else "")
        corr = _overlap(ref.group(1) if ref else "", a)
        return {
            "faithfulness": max(1, round(1 + 4 * faith)),
            "correctness": max(1, round(1 + 4 * corr)),
            "reasoning": "Heuristic token-overlap grade (offline mode).",
        }


ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-5-5"
LLM_KINDS = ("anthropic", *PROVIDERS, "fake")


def detect_llm() -> str:
    """Pick a provider from whichever API key is set; fall back to the offline baseline."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    for kind, p in PROVIDERS.items():
        if p.key_env and os.environ.get(p.key_env):
            return kind
    return "fake"


def get_llm(kind: str, model: str | None = None, max_tokens: int = 1024) -> LLM:
    if kind == "fake":
        return FakeLLM()
    if kind == "anthropic":
        return AnthropicLLM(model=model or ANTHROPIC_DEFAULT_MODEL, max_tokens=max_tokens)
    if kind in PROVIDERS:
        p = PROVIDERS[kind]
        key = os.environ.get(p.key_env) if p.key_env else None
        if p.key_env and not key:
            raise RuntimeError(f"{p.key_env} is not set (needed for --llm {kind})")
        base = os.environ.get("GROUNDWORK_BASE_URL") or p.base_url
        return OpenAICompatLLM(base, model or p.default_model, key, max_tokens=max_tokens)
    raise ValueError(f"Unknown LLM kind {kind!r}; choose from {LLM_KINDS}")
