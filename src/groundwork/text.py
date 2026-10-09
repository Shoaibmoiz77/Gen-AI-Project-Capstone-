"""Shared tokenization used by the lexical index and the hashing embedder."""

from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")

STOPWORDS = frozenset(
    """a an and are as at be by can do does for from has have how i if in is it its of on or
    our s should so that the their there they this to was we what when where which who why
    will with you your my me am""".split()  # noqa: SIM905
)


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    tokens = TOKEN_RE.findall(text.lower())
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return [_light_stem(t) for t in tokens]


def _light_stem(token: str) -> str:
    """A deliberately tiny suffix stripper: enough that 'policies' matches 'policy'."""
    if len(token) <= 4 or token.isdigit():
        return token
    for suffix, repl in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)] + repl
    return token
