from src.index import split_by_tokens


def test_split_short_text_keeps_single_chunk():
    assert split_by_tokens("um texto curto") == ["um texto curto"]


def test_split_long_text_creates_chunks():
    text = "palavra " * 4000
    chunks = split_by_tokens(text, max_tokens=100, overlap=10)
    assert len(chunks) > 1
    assert all(chunks)
