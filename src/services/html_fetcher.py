"""Fetch and parse arXiv papers from ar5iv HTML source."""
import re
import time
from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup, Tag

AR5IV_BASE = "https://ar5iv.labs.arxiv.org/html"
ARXIV_ABS_BASE = "https://arxiv.org/abs"


@dataclass
class Section:
    title: str
    level: int
    content: str


@dataclass
class PaperHTML:
    arxiv_id: str
    title: str
    abstract: str
    sections: list[Section]
    full_text: str
    success: bool
    error: str | None = None


def fetch_html(arxiv_id: str, timeout: float = 30.0) -> str | None:
    """Fetch HTML content from ar5iv."""
    clean_id = arxiv_id.replace("v1", "").replace("v2", "").replace("v3", "")
    url = f"{AR5IV_BASE}/{clean_id}"

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url)
            if response.status_code == 200:
                return response.text
            return None
    except Exception:
        return None


def parse_paper_html(html: str, arxiv_id: str) -> PaperHTML:
    """Parse ar5iv HTML into structured sections."""
    soup = BeautifulSoup(html, "lxml")

    # Extract title
    title_elem = soup.find("h1", class_="ltx_title")
    title = title_elem.get_text(strip=True) if title_elem else ""

    # Extract abstract
    abstract = ""
    abstract_div = soup.find("div", class_="ltx_abstract")
    if abstract_div:
        abstract_p = abstract_div.find("p")
        if abstract_p:
            abstract = abstract_p.get_text(strip=True)

    # Extract sections
    sections = []
    article = soup.find("article") or soup

    for heading in article.find_all(["h2", "h3", "h4"]):
        level = int(heading.name[1])
        section_title = heading.get_text(strip=True)

        # Skip empty or navigation headings
        if not section_title or section_title.lower() in ["contents", "references"]:
            continue

        # Get content until next heading
        content_parts = []
        for sibling in heading.find_next_siblings():
            if isinstance(sibling, Tag):
                if sibling.name in ["h2", "h3", "h4"]:
                    break
                if sibling.name in ["p", "div", "ul", "ol"]:
                    text = sibling.get_text(separator=" ", strip=True)
                    if text:
                        content_parts.append(text)

        if content_parts:
            sections.append(Section(
                title=section_title,
                level=level,
                content=" ".join(content_parts),
            ))

    # Build full text
    full_text_parts = [title, abstract]
    for section in sections:
        full_text_parts.append(f"\n\n## {section.title}\n\n{section.content}")
    full_text = "\n".join(full_text_parts)

    # Clean up text
    full_text = re.sub(r"\s+", " ", full_text)
    full_text = re.sub(r"\n\s*\n+", "\n\n", full_text)

    return PaperHTML(
        arxiv_id=arxiv_id,
        title=title,
        abstract=abstract,
        sections=sections,
        full_text=full_text.strip(),
        success=True,
    )


def fetch_paper_html(arxiv_id: str, timeout: float = 30.0) -> PaperHTML:
    """Fetch and parse a paper from ar5iv."""
    html = fetch_html(arxiv_id, timeout)

    if not html:
        return PaperHTML(
            arxiv_id=arxiv_id,
            title="",
            abstract="",
            sections=[],
            full_text="",
            success=False,
            error="Failed to fetch HTML (may not be available on ar5iv)",
        )

    try:
        return parse_paper_html(html, arxiv_id)
    except Exception as e:
        return PaperHTML(
            arxiv_id=arxiv_id,
            title="",
            abstract="",
            sections=[],
            full_text="",
            success=False,
            error=str(e),
        )


def fetch_papers_html(
    arxiv_ids: list[str],
    delay: float = 1.0,
    timeout: float = 30.0,
) -> list[PaperHTML]:
    """Fetch multiple papers with rate limiting."""
    results = []
    for i, arxiv_id in enumerate(arxiv_ids):
        if i > 0:
            time.sleep(delay)
        results.append(fetch_paper_html(arxiv_id, timeout))
    return results
