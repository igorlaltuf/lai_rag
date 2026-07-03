from src.attachments import excerpt_text, is_obsolete_cached_error


def test_excerpt_text_keeps_short_text_once():
    start, end = excerpt_text("texto curto", edge_chars=20)

    assert start == "texto curto"
    assert end == ""


def test_excerpt_text_uses_start_and_end_for_long_text():
    text = "A" * 50 + "B" * 50 + "C" * 50

    start, end = excerpt_text(text, edge_chars=30)

    assert start == "A" * 30
    assert end == "C" * 30


def test_is_obsolete_cached_error_matches_old_pdf_download_failures():
    assert is_obsolete_cached_error("URL do anexo nao retornou um PDF valido.")
    assert is_obsolete_cached_error("EOF marker not found")
    assert not is_obsolete_cached_error("Arquivo não encontrado")
