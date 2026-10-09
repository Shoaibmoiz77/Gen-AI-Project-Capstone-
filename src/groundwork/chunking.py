"""Structure-aware chunking.

Markdown documents are split on headings first, then packed paragraph by paragraph into
chunks of at most ``max_chars``. Each chunk remembers its heading path ("Expenses > Travel")
and that path is prepended to the text used for retrieval. Short chunks like "Up to $75 per
day" are meaningless alone, but "Expense Policy > Meals > Up to $75 per day" retrieves well.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass(frozen=True)
class Chunk:
    id: str  # "<doc_id>#<n>"
    doc_id: str
    title: str  # document title (first H1, or file stem)
    section: str  # heading path inside the document
    text: str  # the raw chunk text shown to the model

    @property
    def search_text(self) -> str:
        """Text used for indexing: title and section path give short chunks context."""
        return f"{self.title}\n{self.section}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Chunk:
        return cls(**d)


def _split_sections(markdown: str) -> list[tuple[list[str], str]]:
    """Return (heading_path, body) pairs in document order."""
    sections: list[tuple[list[str], str]] = []
    path: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append((list(path), body))
        buf.clear()

    for line in markdown.splitlines():
        m = HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            path[:] = path[: level - 1]
            path.append(m.group(2).strip())
        else:
            buf.append(line)
    flush()
    return sections


def _pack_paragraphs(body: str, max_chars: int) -> list[str]:
    """Greedily pack paragraphs into pieces no longer than max_chars.

    A single paragraph longer than max_chars is split on sentence boundaries.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    units: list[str] = []
    for p in paragraphs:
        if len(p) <= max_chars:
            units.append(p)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", p)
        cur = ""
        for s in sentences:
            if cur and len(cur) + 1 + len(s) > max_chars:
                units.append(cur)
                cur = s
            else:
                cur = f"{cur} {s}".strip()
        if cur:
            units.append(cur)

    pieces: list[str] = []
    cur = ""
    for u in units:
        if cur and len(cur) + 2 + len(u) > max_chars:
            pieces.append(cur)
            cur = u
        else:
            cur = f"{cur}\n\n{u}" if cur else u
    if cur:
        pieces.append(cur)
    return pieces


def chunk_document(doc_id: str, text: str, max_chars: int = 900) -> list[Chunk]:
    sections = _split_sections(text)
    title = doc_id
    for path, _ in sections:
        if path:
            title = path[0]
            break
    else:
        m = HEADING_RE.match(text.strip().splitlines()[0]) if text.strip() else None
        if m:
            title = m.group(2).strip()

    chunks: list[Chunk] = []
    for path, body in sections:
        section = " > ".join(path[1:]) if len(path) > 1 else (path[0] if path else "")
        for piece in _pack_paragraphs(body, max_chars):
            chunks.append(
                Chunk(
                    id=f"{doc_id}#{len(chunks)}",
                    doc_id=doc_id,
                    title=title,
                    section=section,
                    text=piece,
                )
            )
    return chunks


def load_corpus(src: str | Path, max_chars: int = 900) -> list[Chunk]:
    """Chunk every .md and .txt file under ``src``. The doc_id is the file stem."""
    src = Path(src)
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in {".md", ".txt"})
    if not files:
        raise FileNotFoundError(f"No .md or .txt files found under {src}")
    chunks: list[Chunk] = []
    for f in files:
        chunks.extend(chunk_document(f.stem, f.read_text(encoding="utf-8"), max_chars))
    return chunks
