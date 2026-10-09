from groundwork.chunking import chunk_document

DOC = """# Handbook

Intro paragraph.

## Leave

### Sick Leave

Ten days per year.

## Travel

""" + "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(6))


def test_heading_path_and_title():
    chunks = chunk_document("hb", DOC, max_chars=400)
    assert all(c.title == "Handbook" for c in chunks)
    sick = next(c for c in chunks if "Ten days" in c.text)
    assert sick.section == "Leave > Sick Leave"
    assert "Leave > Sick Leave" in sick.search_text


def test_chunks_respect_size_and_have_unique_ids():
    chunks = chunk_document("hb", DOC, max_chars=400)
    assert all(len(c.text) <= 400 for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)
    assert [c.id for c in chunks][:2] == ["hb#0", "hb#1"]


def test_long_paragraph_is_split_on_sentences():
    long_para = " ".join(f"Sentence number {i} is here." for i in range(100))
    chunks = chunk_document("x", f"# T\n\n{long_para}", max_chars=200)
    assert len(chunks) > 5
    assert all(len(c.text) <= 200 for c in chunks)
    assert all(c.text.endswith(".") for c in chunks)


def test_corpus_loads(chunks):
    docs = {c.doc_id for c in chunks}
    assert {"time-off-policy", "expense-policy", "kestrel-x2-faq"} <= docs
