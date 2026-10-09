"""LLM clients.

Every model call in this project asks for a *forced tool call* rather than free text. The
tool's JSON schema is the output contract, so we get structured answers (sentences, citation
ids, abstention flags, judge scores) without fragile regex parsing of prose.

``FakeLLM`` implements the same interface deterministically, so tests and the offline eval
run with no API key and no network.
"""

from __future__ import annotations

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
            temperature=0,
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


def get_llm(kind: str, model: str, max_tokens: int = 1024) -> LLM:
    if kind == "fake":
        return FakeLLM()
    if kind == "anthropic":
        return AnthropicLLM(model=model, max_tokens=max_tokens)
    raise ValueError(f"Unknown LLM kind {kind!r}")
