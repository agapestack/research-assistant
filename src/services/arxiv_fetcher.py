"""arXiv paper search utilities."""
import arxiv


def search_papers(
    query: str,
    max_results: int = 10,
    sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance,
) -> list[arxiv.Result]:
    """Search arxiv for papers matching query."""
    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=max_results, sort_by=sort_by)
    return list(client.results(search))
