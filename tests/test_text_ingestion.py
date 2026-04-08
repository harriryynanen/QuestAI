import shutil
import uuid
from pathlib import Path

import pytest

from app.retrieval.chunker import MarkdownChunker
from app.retrieval.document_store import DocumentStore


@pytest.fixture
def workspace_tmp_text_dir() -> Path:
    temp_dir = Path("tests") / "_tmp_text" / uuid.uuid4().hex
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_txt_document_is_loaded_into_retrieval_documents(workspace_tmp_text_dir):
    text_path = workspace_tmp_text_dir / "notes.txt"
    text_path.write_text("Tax arrears should be escalated for manual review.", encoding="utf-8")

    store = DocumentStore(workspace_tmp_text_dir)
    bundle = store.load_retrieval_bundle()

    assert len(bundle.documents) == 1
    assert bundle.documents[0].file_name == "notes.txt"
    assert bundle.documents[0].source_type == "text"
    assert bundle.issues == []


def test_txt_document_enters_same_chunking_flow(workspace_tmp_text_dir):
    text_path = workspace_tmp_text_dir / "notes.txt"
    text_path.write_text(
        "Tax arrears should be escalated for manual review.\n\nPayment delays require caution.",
        encoding="utf-8",
    )

    store = DocumentStore(workspace_tmp_text_dir)
    documents = store.load_retrieval_documents()
    chunks = MarkdownChunker(max_characters=120).chunk_documents(documents)

    assert any(document.source_type == "text" for document in documents)
    assert any(chunk.file_name == "notes.txt" for chunk in chunks)


def test_supported_document_types_include_txt_for_discovery_and_retrieval(workspace_tmp_text_dir):
    markdown_path = workspace_tmp_text_dir / "guide.md"
    markdown_path.write_text("# Policy\nPolicy text.\n", encoding="utf-8")
    text_path = workspace_tmp_text_dir / "notes.txt"
    text_path.write_text("Plain text guidance.", encoding="utf-8")
    pdf_path = workspace_tmp_text_dir / "notes.pdf"
    pdf_path.write_bytes(b"%PDF-test")

    store = DocumentStore(workspace_tmp_text_dir)
    listed_extensions = {document.extension for document in store.list_documents()}

    assert listed_extensions == {".md", ".txt", ".pdf"}
