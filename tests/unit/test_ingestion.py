from app.ingestion.chunking import chunk_documents
from app.ingestion.cleaning import clean_text
from app.ingestion.loaders import load_directory, load_file
from tests.conftest import FIXTURES, make_pdf


def test_clean_text_normalizes():
    raw = "Hola   mundo\r\n\r\n\r\n\r\nconfigu-\nración\x00"
    assert clean_text(raw) == "Hola mundo\n\nconfiguración"


def test_load_markdown_and_txt():
    docs = load_directory(FIXTURES)
    types = {d.metadata["file_type"] for d in docs}
    assert types == {"md", "txt"}
    assert all(d.metadata["doc_id"] for d in docs)


def test_load_pdf_has_page(tmp_path):
    p = tmp_path / "x.pdf"
    p.write_bytes(make_pdf("Hello PDF world"))
    docs = load_file(p)
    assert docs[0].metadata["page"] == 1
    assert "Hello PDF world" in docs[0].page_content


def test_markdown_chunks_keep_section():
    docs = load_file(FIXTURES / "deploy.md")
    chunks = chunk_documents(docs, 200, 20)
    assert any(
        c.metadata.get("section") == "Guía de despliegue > Variables de entorno" for c in chunks
    )


def test_chunk_ids_are_deterministic():
    docs = load_directory(FIXTURES)
    a = [c.metadata["chunk_id"] for c in chunk_documents(docs, 200, 20)]
    b = [c.metadata["chunk_id"] for c in chunk_documents(load_directory(FIXTURES), 200, 20)]
    assert a == b and len(set(a)) == len(a)
