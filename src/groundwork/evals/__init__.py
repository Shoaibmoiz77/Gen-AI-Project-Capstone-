"""Evaluation harness: retrieval metrics, citation checks and LLM-as-judge grading."""

from groundwork.evals.runner import EvalCase, load_golden, run_eval

__all__ = ["EvalCase", "load_golden", "run_eval"]
