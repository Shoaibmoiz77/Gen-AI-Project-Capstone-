"""LLM-as-judge grading of answers.

Two separate scores, because they fail independently:

* faithfulness - is every claim supported by the sources the answer cited? (hallucination)
* correctness  - does the answer match the reference answer? (usefulness)

An answer can be perfectly faithful and still wrong (it cited the wrong policy), or correct
but unfaithful (the model knew the answer and cited something irrelevant). Tracking both
tells you whether to fix retrieval or generation.
"""

from __future__ import annotations

from dataclasses import dataclass

from groundwork.llm import LLM, Usage

JUDGE_SYSTEM = """You are a strict grader for a question-answering system that must answer \
only from cited sources. Grade on two independent 1-5 scales.

faithfulness: Is every statement in the answer directly supported by the cited sources?
  5 = every claim supported; 3 = mostly supported with a minor unsupported detail;
  1 = key claims are unsupported or contradicted by the sources.
correctness: Does the answer convey the same facts as the reference answer?
  5 = all key facts present and accurate; 3 = partially correct or missing a key fact;
  1 = wrong or does not answer the question.

Judge content, not wording or length. Be concise in your reasoning."""

GRADE_TOOL = {
    "name": "submit_grade",
    "description": "Submit the grades.",
    "input_schema": {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string", "description": "One or two sentences."},
            "faithfulness": {"type": "integer", "minimum": 1, "maximum": 5},
            "correctness": {"type": "integer", "minimum": 1, "maximum": 5},
        },
        "required": ["reasoning", "faithfulness", "correctness"],
    },
}


@dataclass
class Grade:
    faithfulness: int
    correctness: int
    reasoning: str
    usage: Usage


def _clamp(v: object) -> int:
    try:
        return max(1, min(5, int(v)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 1


def grade_answer(
    judge: LLM, question: str, reference: str, answer: str, cited_texts: list[str]
) -> Grade:
    sources = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(cited_texts, start=1)) or "(none)"
    user = (
        f"<question>\n{question}\n</question>\n\n"
        f"<reference_answer>\n{reference}\n</reference_answer>\n\n"
        f"<answer>\n{answer}\n</answer>\n\n"
        f"<cited_sources>\n{sources}\n</cited_sources>"
    )
    r = judge.call_tool(JUDGE_SYSTEM, user, GRADE_TOOL)
    return Grade(
        faithfulness=_clamp(r.input.get("faithfulness")),
        correctness=_clamp(r.input.get("correctness")),
        reasoning=str(r.input.get("reasoning", "")),
        usage=r.usage,
    )
