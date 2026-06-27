"""
arXiv search tool for the research agent.
Uses the arXiv API to fetch recent papers on a topic.
"""
from __future__ import annotations
import urllib.request, urllib.parse, xml.etree.ElementTree as ET
from typing import List, Dict, Optional
from dataclasses import dataclass
import time, logging

logger = logging.getLogger(__name__)

ARXIV_API = "http://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str]
    published: str
    url: str
    categories: List[str]
    pdf_url: str = ""


class ArxivSearchTool:
    """
    Search arXiv for papers matching a query.
    Respects arXiv's 3-second rate limit between requests.
    """

    def __init__(self, max_results: int = 10, sort_by: str = "submittedDate"):
        self.max_results = max_results
        self.sort_by = sort_by
        self._last_request = 0

    def search(self, query: str, max_results: Optional[int] = None) -> List[ArxivPaper]:
        """Search arXiv and return structured results."""
        self._rate_limit()
        n = max_results or self.max_results
        url = self._build_url(query, n)

        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                xml_data = resp.read()
        except Exception as e:
            logger.error(f"arXiv request failed: {e}")
            return []

        return self._parse(xml_data)

    def search_recent(self, query: str, days: int = 30) -> List[ArxivPaper]:
        """Search for papers submitted in the last N days."""
        papers = self.search(query, max_results=50)
        from datetime import datetime, timedelta, timezone
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        return [
            p for p in papers
            if self._parse_date(p.published) >= cutoff
        ]

    def get_by_id(self, arxiv_id: str) -> Optional[ArxivPaper]:
        """Fetch a specific paper by arXiv ID."""
        results = self.search(f"id:{arxiv_id}", max_results=1)
        return results[0] if results else None

    def _build_url(self, query: str, n: int) -> str:
        params = urllib.parse.urlencode({
            "search_query": f"all:{query}",
            "max_results": n,
            "sortBy": self.sort_by,
            "sortOrder": "descending",
        })
        return f"{ARXIV_API}?{params}"

    def _parse(self, xml_data: bytes) -> List[ArxivPaper]:
        root = ET.fromstring(xml_data)
        papers = []
        for entry in root.findall("atom:entry", NS):
            try:
                arxiv_id = entry.find("atom:id", NS).text.split("/abs/")[-1]
                title = entry.find("atom:title", NS).text.strip().replace("\n", " ")
                abstract = entry.find("atom:summary", NS).text.strip()
                published = entry.find("atom:published", NS).text[:10]
                url = entry.find("atom:id", NS).text
                authors = [
                    a.find("atom:name", NS).text
                    for a in entry.findall("atom:author", NS)
                ]
                categories = [
                    c.get("term", "") for c in entry.findall("arxiv:primary_category", NS)
                ] + [c.get("term", "") for c in entry.findall("atom:category", NS)]

                papers.append(ArxivPaper(
                    arxiv_id=arxiv_id, title=title, abstract=abstract,
                    authors=authors, published=published, url=url,
                    categories=list(set(categories)),
                    pdf_url=url.replace("/abs/", "/pdf/") + ".pdf",
                ))
            except Exception as e:
                logger.debug(f"Failed to parse entry: {e}")
        return papers

    def _rate_limit(self):
        elapsed = time.time() - self._last_request
        if elapsed < 3.0:
            time.sleep(3.0 - elapsed)
        self._last_request = time.time()

    @staticmethod
    def _parse_date(date_str: str):
        from datetime import datetime, timezone
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            return datetime.min.replace(tzinfo=timezone.utc)


def format_for_context(papers: List[ArxivPaper], max_per_paper: int = 600) -> str:
    """Format papers as context string for LLM."""
    lines = []
    for i, p in enumerate(papers, 1):
        snippet = p.abstract[:max_per_paper] + ("..." if len(p.abstract) > max_per_paper else "")
        lines.append(
            f"[{i}] {p.title} ({p.published})\n"
            f"Authors: {', '.join(p.authors[:3])}{'et al.' if len(p.authors) > 3 else ''}\n"
            f"URL: {p.url}\n"
            f"Abstract: {snippet}\n"
        )
    return "\n".join(lines)
