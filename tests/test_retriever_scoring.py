from pathlib import Path

from models import DocumentChunk, DocumentRecord
from retrieval.chunker import MarkdownChunker
from retrieval.retriever import KeywordRetriever


def test_retriever_prefers_exact_phrase_and_heading_matches():
    retriever = KeywordRetriever(top_k=2)
    chunks = [
        DocumentChunk(
            chunk_id="chunk-1",
            document_id="policy",
            file_name="policy.md",
            text="Repeated payment delays may require escalation and manual review.",
            section_heading="Operational notes",
        ),
        DocumentChunk(
            chunk_id="chunk-2",
            document_id="policy",
            file_name="policy.md",
            text="Unresolved tax arrears block a confident positive preliminary view.",
            section_heading="Tax arrears guidance",
        ),
    ]

    results = retriever.retrieve("What does the policy say about tax arrears?", chunks)

    assert results
    assert results[0].chunk.chunk_id == "chunk-2"
    assert "phrase match: tax arrears" in results[0].match_summary
    assert "heading match: tax, arrear" in results[0].match_summary


def test_retriever_can_expand_generic_product_questions_to_product_sections():
    chunker = MarkdownChunker(max_characters=400)
    document = DocumentRecord(
        document_id="product_guide",
        file_name="demo_product_guide.md",
        path=Path("demo_product_guide.md"),
        text=(
            "# Demo Product Guide\n"
            "## Overview\n"
            "This guide describes three fictional financing products used in a lightweight business banking advisory demo.\n\n"
            "### FlexLine Demo\n"
            "Purpose: Short-term working capital flexibility.\n\n"
            "### InvoiceBridge Demo\n"
            "Purpose: Financing against eligible business-to-business receivables.\n\n"
            "### AssetGrow Demo\n"
            "Purpose: Equipment and machinery investment support.\n"
        ),
        source_type="markdown",
    )
    chunks = chunker.chunk_documents([document])
    retriever = KeywordRetriever(top_k=3)

    results = retriever.retrieve("What products are mentioned in the guide?", chunks)

    assert results
    assert any(
        any(
            product in f"{item.chunk.section_heading or ''} {item.chunk.text}".lower()
            for product in ("flexline", "invoicebridge", "assetgrow")
        )
        for item in results
    )
    assert any("query expansion:" in item.match_summary for item in results)
