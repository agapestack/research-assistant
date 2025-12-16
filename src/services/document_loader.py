from pathlib import Path

import fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .html_fetcher import PaperHTML, fetch_paper_html


def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """Extract text from PDF with page metadata."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():
            pages.append({"text": text, "page": page_num})
    doc.close()
    return pages


def extract_text_from_html(paper_html: PaperHTML) -> list[dict]:
    """Convert PaperHTML sections to page-like structure for chunking."""
    if not paper_html.success:
        return []

    pages = []
    # Treat each section as a "page" for chunking purposes
    if paper_html.abstract:
        pages.append({"text": f"Abstract\n\n{paper_html.abstract}", "page": 0})

    for i, section in enumerate(paper_html.sections, start=1):
        text = f"{section.title}\n\n{section.content}"
        pages.append({"text": text, "page": i})

    return pages


def chunk_document(
    pages: list[dict],
    source: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    extra_metadata: dict | None = None,
) -> list[Document]:
    """Split pages into chunks with metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents = []
    for page_data in pages:
        chunks = splitter.split_text(page_data["text"])
        for i, chunk in enumerate(chunks):
            metadata = {
                "source": source,
                "page": page_data["page"],
                "chunk": i,
            }
            if extra_metadata:
                metadata.update(extra_metadata)
            documents.append(Document(page_content=chunk, metadata=metadata))
    return documents


def load_pdf(
    pdf_path: Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    extra_metadata: dict | None = None,
) -> list[Document]:
    """Load a PDF and return chunked documents."""
    pages = extract_text_from_pdf(pdf_path)
    return chunk_document(
        pages,
        source=str(pdf_path),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        extra_metadata=extra_metadata,
    )


def load_papers(
    papers: list[dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Load papers with arxiv metadata (PDF mode)."""
    documents = []
    for paper in papers:
        extra_metadata = {
            "title": paper["title"],
            "authors": ", ".join(paper["authors"])
            if isinstance(paper["authors"], list)
            else paper["authors"],
            "published": paper["published"],
            "journal": paper.get("journal"),
            "categories": ", ".join(paper.get("categories", []))
            if isinstance(paper.get("categories"), list)
            else paper.get("categories"),
            "arxiv_url": paper["arxiv_url"],
            "arxiv_id": paper.get("id") or paper.get("arxiv_id"),
        }
        documents.extend(
            load_pdf(
                paper["pdf_path"],
                chunk_size,
                chunk_overlap,
                extra_metadata,
            )
        )
    return documents


def load_paper_from_html(
    arxiv_id: str,
    metadata: dict,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Load a paper from ar5iv HTML source."""
    paper_html = fetch_paper_html(arxiv_id)

    if not paper_html.success:
        return []

    pages = extract_text_from_html(paper_html)
    if not pages:
        return []

    extra_metadata = {
        "title": paper_html.title or metadata.get("title", ""),
        "authors": metadata.get("authors", ""),
        "published": metadata.get("published", ""),
        "categories": metadata.get("categories", ""),
        "arxiv_url": metadata.get("arxiv_url", f"https://arxiv.org/abs/{arxiv_id}"),
        "arxiv_id": arxiv_id,
        "source_type": "html",
    }

    return chunk_document(
        pages,
        source=f"ar5iv:{arxiv_id}",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        extra_metadata=extra_metadata,
    )


def load_papers_from_html(
    papers: list[dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Load multiple papers from ar5iv HTML source."""
    documents = []
    for paper in papers:
        arxiv_id = paper.get("arxiv_id") or paper.get("id")
        metadata = {
            "title": paper.get("title", ""),
            "authors": ", ".join(paper["authors"])
            if isinstance(paper.get("authors"), list)
            else paper.get("authors", ""),
            "published": paper.get("published", ""),
            "categories": ", ".join(paper["categories"])
            if isinstance(paper.get("categories"), list)
            else paper.get("categories", ""),
            "arxiv_url": paper.get("arxiv_url", ""),
        }
        docs = load_paper_from_html(arxiv_id, metadata, chunk_size, chunk_overlap)
        documents.extend(docs)
    return documents
