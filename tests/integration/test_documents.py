import pytest

from app.dependencies import get_document_service
from app.documents.service import DocumentService
from tests.conftest import make_pdf


@pytest.fixture
def docs_client(client, vectorstore, settings, tmp_path):
    s = settings.model_copy(update={"docs_dir": tmp_path / "docs", "max_upload_mb": 1})
    client.app.dependency_overrides[get_document_service] = lambda: DocumentService(vectorstore, s)
    client.docs_dir = s.docs_dir
    return client


def upload(client, name, content, ctype="text/plain"):
    return client.post("/documents", files={"file": (name, content, ctype)})


def test_list_includes_preingested_fixtures(docs_client):
    body = docs_client.get("/documents").json()
    assert body["total"] == len(body["documents"]) >= 2
    assert {d["source"] for d in body["documents"]} >= {"cache.txt", "deploy.md"}


def test_upload_get_and_delete_roundtrip(docs_client):
    r = upload(docs_client, "nuevo.md", b"# Titulo\n\nContenido unico sobre kubernetes.")
    assert r.status_code == 201
    doc = r.json()
    assert doc["source"] == "nuevo.md" and doc["chunks"] >= 1 and doc["file_type"] == "md"
    assert (docs_client.docs_dir / "nuevo.md").exists()

    assert docs_client.get(f"/documents/{doc['doc_id']}").json()["doc_id"] == doc["doc_id"]

    assert docs_client.delete(f"/documents/{doc['doc_id']}").status_code == 204
    assert docs_client.get(f"/documents/{doc['doc_id']}").status_code == 404
    assert not (docs_client.docs_dir / "nuevo.md").exists()


def test_upload_pdf_reports_pages(docs_client):
    r = upload(docs_client, "a.pdf", make_pdf("Hola mundo PDF"), "application/pdf")
    assert r.status_code == 201
    assert r.json()["pages"] == 1


def test_duplicate_content_returns_409(docs_client):
    assert upload(docs_client, "uno.txt", b"texto repetido").status_code == 201
    r = upload(docs_client, "otro.txt", b"texto repetido")
    assert r.status_code == 409
    assert r.json()["error"] == "duplicate_document"


def test_same_name_new_content_replaces_old(docs_client):
    first = upload(docs_client, "v.txt", b"version uno").json()
    second = upload(docs_client, "v.txt", b"version dos").json()
    assert first["doc_id"] != second["doc_id"]
    assert docs_client.get(f"/documents/{first['doc_id']}").status_code == 404
    sources = [d["source"] for d in docs_client.get("/documents").json()["documents"]]
    assert sources.count("v.txt") == 1


def test_unsupported_extension_returns_422(docs_client):
    r = upload(docs_client, "x.exe", b"MZ")
    assert r.status_code == 422
    assert r.json()["error"] == "invalid_document"


def test_empty_text_returns_422_and_leaves_no_file(docs_client):
    r = upload(docs_client, "vacio.txt", b"   \n  ")
    assert r.status_code == 422
    assert not (docs_client.docs_dir / "vacio.txt").exists()


def test_path_traversal_filename_is_sanitized(docs_client):
    r = upload(docs_client, "../../evil.txt", b"contenido inocente")
    assert r.status_code == 201
    assert r.json()["source"] == "evil.txt"
    assert (docs_client.docs_dir / "evil.txt").exists()


def test_too_large_returns_413(docs_client):
    r = upload(docs_client, "big.txt", b"a" * (1024 * 1024 + 1))
    assert r.status_code == 413
    assert r.json()["error"] == "document_too_large"


def test_delete_unknown_returns_404(docs_client):
    r = docs_client.delete("/documents/noexiste")
    assert r.status_code == 404
    assert r.json()["error"] == "document_not_found"
