from pathlib import Path

import pytest

from src.services.document_loader import chunk_document, extract_text_from_pdf


class TestChunkDocument:
    def test_chunk_splits_text(self):
        pages = [{"text": "This is a test. " * 100, "page": 1}]
        chunks = chunk_document(pages, source="test.pdf", extra_metadata={"title": "Test"})

        assert len(chunks) > 0
        assert all(hasattr(c, "page_content") for c in chunks)
        assert all(hasattr(c, "metadata") for c in chunks)

    def test_chunk_preserves_metadata(self):
        pages = [{"text": "Short text for testing chunking.", "page": 1}]
        extra_metadata = {"title": "Test Paper", "arxiv_url": "http://example.com"}
        chunks = chunk_document(pages, source="test.pdf", extra_metadata=extra_metadata)

        assert chunks[0].metadata["title"] == "Test Paper"
        assert chunks[0].metadata["arxiv_url"] == "http://example.com"

    def test_chunk_adds_source_and_page(self):
        pages = [{"text": "Test content for chunking.", "page": 5}]
        chunks = chunk_document(pages, source="test.pdf")

        assert chunks[0].metadata["source"] == "test.pdf"
        assert chunks[0].metadata["page"] == 5
        assert "chunk" in chunks[0].metadata

    def test_chunk_handles_multiple_pages(self):
        pages = [
            {"text": "Page one content.", "page": 1},
            {"text": "Page two content.", "page": 2},
        ]
        chunks = chunk_document(pages, source="test.pdf")

        assert len(chunks) == 2
        assert chunks[0].metadata["page"] == 1
        assert chunks[1].metadata["page"] == 2


class TestExtractTextFromPdf:
    def test_extract_text_handles_missing_file(self):
        with pytest.raises(Exception):
            extract_text_from_pdf(Path("/nonexistent/file.pdf"))
