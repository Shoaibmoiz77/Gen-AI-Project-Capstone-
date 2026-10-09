import json
from types import SimpleNamespace

import pytest

from groundwork import llm as llm_mod
from groundwork.generation import ANSWER_TOOL, generate_answer
from groundwork.llm import OpenAICompatLLM, detect_llm, get_llm


class FakeHTTP:
    """Returns queued (status, body, headers) responses and records requests."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def post(self, url, json=None):
        self.requests.append((url, json))
        status, body, headers = self.responses.pop(0)
        return SimpleNamespace(
            status_code=status,
            headers=headers,
            json=lambda: body,
            text=str(body),
        )


def _ok(args: dict):
    return (
        200,
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"function": {"name": "submit_answer", "arguments": json.dumps(args)}}
                        ]
                    }
                }
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 7},
        },
        {},
    )


def test_openai_compatible_payload_and_parsing(retriever):
    http = FakeHTTP([_ok({"answerable": True, "sentences": [{"text": "Yes.", "citations": [1]}]})])
    llm = OpenAICompatLLM("https://example.test/v1/", "some-model", "key", client=http)
    ans = generate_answer(llm, "q?", retriever.search("password", k=2))

    url, payload = http.requests[0]
    assert url == "https://example.test/v1/chat/completions"
    assert payload["messages"][0]["role"] == "system"
    assert payload["tools"][0]["function"]["parameters"] == ANSWER_TOOL["input_schema"]
    assert payload["tool_choice"] == {"type": "function", "function": {"name": "submit_answer"}}
    assert ans.render() == "Yes. [1]"
    assert ans.usage.input_tokens == 50


def test_rate_limit_is_retried(retriever, monkeypatch):
    sleeps = []
    monkeypatch.setattr(llm_mod.time, "sleep", sleeps.append)
    http = FakeHTTP(
        [
            (429, {"error": "slow down"}, {"retry-after": "3"}),
            (503, {"error": "busy"}, {}),
            _ok({"answerable": False, "sentences": []}),
        ]
    )
    llm = OpenAICompatLLM("https://x.test/v1", "m", client=http)
    ans = generate_answer(llm, "q?", retriever.search("password", k=2))
    assert not ans.answerable
    assert sleeps == [3.0, 4.0]  # honours Retry-After, then exponential backoff


def test_client_errors_are_not_retried(retriever):
    http = FakeHTTP([(401, {"error": "bad key"}, {})])
    llm = OpenAICompatLLM("https://x.test/v1", "m", client=http)
    with pytest.raises(RuntimeError, match="401"):
        generate_answer(llm, "q?", retriever.search("password", k=2))
    assert len(http.requests) == 1


def test_provider_detection(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert detect_llm() == "fake"
    monkeypatch.setenv("GROQ_API_KEY", "x")
    assert detect_llm() == "groq"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "y")
    assert detect_llm() == "anthropic"


def test_missing_key_gives_clear_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        get_llm("gemini")


def test_ollama_needs_no_key():
    llm = get_llm("ollama")
    assert llm.name == "llama3.1"


def test_unknown_model_lists_alternatives():
    class HTTP(FakeHTTP):
        def get(self, url):
            assert url == "https://x.test/v1/models"
            return SimpleNamespace(json=lambda: {"data": [{"id": "b-model"}, {"id": "a-model"}]})

    http = HTTP([(404, {"error": {"message": "The model `old` does not exist"}}, {})])
    llm = OpenAICompatLLM("https://x.test/v1", "old", client=http)
    with pytest.raises(RuntimeError, match="a-model, b-model"):
        llm.call_tool("s", "u", ANSWER_TOOL)


def test_malformed_tool_output_is_resampled():
    bad = (400, {"error": {"code": "output_parse_failed"}}, {})
    http = FakeHTTP([bad, bad, _ok({"answerable": False, "sentences": []})])
    llm = OpenAICompatLLM("https://x.test/v1", "m", client=http)
    llm.call_tool("s", "u", ANSWER_TOOL)
    assert len(http.requests) == 3


def test_eval_survives_generation_errors(pipeline, tmp_path):
    from pathlib import Path

    from groundwork.evals import run_eval

    class Flaky:
        name = "flaky"

        def __init__(self):
            self.inner, self.n = llm_mod.FakeLLM(), 0

        def call_tool(self, system, user, tool):
            self.n += 1
            if self.n % 5 == 0:
                raise RuntimeError("boom")
            return self.inner.call_tool(system, user, tool)

    pipeline.llm = Flaky()
    golden = Path(__file__).resolve().parents[1] / "data" / "eval" / "golden.jsonl"
    s = run_eval(pipeline, golden, tmp_path, judge=None, workers=1)["summary"]
    assert s["n_errors"] > 0
    assert s["retrieval"]["recall@6"] >= 0.9  # retrieval still scored for every case
    assert "## Errors" in (tmp_path / "report.md").read_text()
