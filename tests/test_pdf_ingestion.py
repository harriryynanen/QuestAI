import shutil
import uuid
from pathlib import Path

import pytest

from retrieval.chunker import MarkdownChunker
from retrieval.document_store import DocumentStore


@pytest.fixture
def workspace_tmp_dir() -> Path:
    temp_dir = Path("tests") / "_tmp_pdf" / uuid.uuid4().hex
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_pdf_document_is_loaded_when_text_extraction_succeeds(workspace_tmp_dir, monkeypatch):
    pdf_path = workspace_tmp_dir / "policy.pdf"
    pdf_path.write_bytes(b"%PDF-test")

    store = DocumentStore(workspace_tmp_dir, pdf_min_text_characters=10)
    monkeypatch.setattr(
        store,
        "_extract_pdf_text",
        lambda path: "## Page 1\nTax arrears should be treated cautiously.",
    )

    bundle = store.load_retrieval_bundle()
    documents = bundle.documents
    issues = bundle.issues

    assert len(documents) == 1
    assert documents[0].file_name == "policy.pdf"
    assert documents[0].source_type == "pdf"
    assert "Tax arrears" in documents[0].text
    assert issues == []


def test_empty_pdf_is_skipped_gracefully(workspace_tmp_dir, monkeypatch):
    pdf_path = workspace_tmp_dir / "empty.pdf"
    pdf_path.write_bytes(b"%PDF-test")

    store = DocumentStore(workspace_tmp_dir, pdf_min_text_characters=20)
    monkeypatch.setattr(store, "_extract_pdf_text", lambda path: "too short")

    bundle = store.load_retrieval_bundle()
    documents = bundle.documents
    issues = bundle.issues

    assert documents == []
    assert len(issues) == 1
    assert issues[0].file_name == "empty.pdf"
    assert "too little usable text" in issues[0].reason.lower()


def test_mixed_markdown_and_pdf_documents_enter_same_retrieval_flow(workspace_tmp_dir, monkeypatch):
    markdown_path = workspace_tmp_dir / "guide.md"
    markdown_path.write_text("# Policy\nTax arrears should lead to caution.\n", encoding="utf-8")
    pdf_path = workspace_tmp_dir / "notes.pdf"
    pdf_path.write_bytes(b"%PDF-test")

    store = DocumentStore(workspace_tmp_dir, pdf_min_text_characters=10)
    monkeypatch.setattr(
        store,
        "_extract_pdf_text",
        lambda path: "## Page 1\nAdvisor note: unresolved tax arrears should be escalated.",
    )

    documents = store.load_retrieval_documents()
    chunks = MarkdownChunker(max_characters=200).chunk_documents(documents)

    assert len(documents) == 2
    assert any(document.source_type == "markdown" for document in documents)
    assert any(document.source_type == "pdf" for document in documents)
    assert any(chunk.file_name == "notes.pdf" for chunk in chunks)
    assert any(chunk.section_heading == "Page 1" for chunk in chunks)


def test_retrieval_bundle_loads_documents_and_issues_in_one_pass(workspace_tmp_dir, monkeypatch):
    markdown_path = workspace_tmp_dir / "guide.md"
    markdown_path.write_text("# Policy\nPolicy text.\n", encoding="utf-8")
    pdf_path = workspace_tmp_dir / "notes.pdf"
    pdf_path.write_bytes(b"%PDF-test")

    store = DocumentStore(workspace_tmp_dir, pdf_min_text_characters=10)
    call_count = {"count": 0}

    def fake_extract(path):
        call_count["count"] += 1
        return "## Page 1\nPDF policy text."

    monkeypatch.setattr(store, "_extract_pdf_text", fake_extract)

    bundle = store.load_retrieval_bundle()

    assert len(bundle.documents) == 2
    assert bundle.issues == []
    assert call_count["count"] == 1


def test_list_documents_matches_mvp_supported_types(workspace_tmp_dir):
    markdown_path = workspace_tmp_dir / "guide.md"
    markdown_path.write_text("# Policy\nPolicy text.\n", encoding="utf-8")
    pdf_path = workspace_tmp_dir / "notes.pdf"
    pdf_path.write_bytes(b"%PDF-test")

    store = DocumentStore(workspace_tmp_dir)
    listed_extensions = {document.extension for document in store.list_documents()}

    assert listed_extensions == {".md", ".pdf"}
