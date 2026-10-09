from types import SimpleNamespace

from groundwork.generation import (
    ABSTAIN_TEXT,
    ANSWER_TOOL,
    build_user_prompt,
    generate_answer,
    parse_answer,
)
from groundwork.llm import AnthropicLLM, ToolResult


def test_prompt_numbers_sources(retriever):
    hits = retriever.search("hotel limit London", k=3)
    prompt = build_user_prompt("hotel limit?", hits)
    assert '<source id="1"' in prompt and '<source id="3"' in prompt
    assert "<question>\nhotel limit?\n</question>" in prompt


def test_parse_answer_maps_ids_and_drops_invalid_citations(retriever):
    hits = retriever.search("hotel limit London", k=3)
    result = ToolResult(
        input={
            "answerable": True,
            "sentences": [
                {"text": "Hotels in London are capped at $350.", "citations": [1, 1, 99]},
                {"text": "This sentence has no valid source.", "citations": ["x"]},
            ],
        }
    )
    ans = parse_answer(result, hits)
    assert ans.sentences[0].chunk_ids == [hits[0].chunk.id]
    assert ans.dropped_citations == 2
    assert not ans.sentences[1].supported
    assert ans.render().startswith("Hotels in London are capped at $350. [1]")


def test_abstention_renders_fallback(retriever):
    hits = retriever.search("anything", k=2)
    ans = parse_answer(ToolResult(input={"answerable": False, "sentences": []}), hits)
    assert not ans.answerable
    assert ans.render() == ABSTAIN_TEXT


def test_answerable_true_with_no_sentences_is_abstention(retriever):
    hits = retriever.search("anything", k=2)
    ans = parse_answer(ToolResult(input={"answerable": True, "sentences": []}), hits)
    assert not ans.answerable


def test_fake_llm_end_to_end(pipeline):
    res = pipeline.ask("How many sick days does each employee receive per year?")
    assert res.answer.answerable
    assert "10 paid sick days" in res.answer.render()
    assert res.sources()[0]["doc_id"] == "time-off-policy"


def test_anthropic_client_forces_tool_call(retriever):
    calls = {}

    class FakeMessages:
        def create(self, **kw):
            calls.update(kw)
            return SimpleNamespace(
                content=[
                    SimpleNamespace(type="text", text="thinking..."),
                    SimpleNamespace(
                        type="tool_use",
                        name="submit_answer",
                        input={
                            "answerable": True,
                            "sentences": [{"text": "Yes.", "citations": [1]}],
                        },
                    ),
                ],
                usage=SimpleNamespace(input_tokens=120, output_tokens=30),
            )

    llm = AnthropicLLM(model="test-model", client=SimpleNamespace(messages=FakeMessages()))
    ans = generate_answer(llm, "q?", retriever.search("password", k=2))
    assert calls["tool_choice"] == {"type": "tool", "name": "submit_answer"}
    assert calls["tools"] == [ANSWER_TOOL]
    assert calls["temperature"] == 0
    assert ans.usage.input_tokens == 120
    assert ans.render() == "Yes. [1]"
