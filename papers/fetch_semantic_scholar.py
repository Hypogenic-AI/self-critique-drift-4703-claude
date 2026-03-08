"""Fetch papers from Semantic Scholar API for representation drift research.

Uses aggressive retry/backoff to handle rate limiting on the free tier.
"""

import json
import time
import requests

API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,authors,year,abstract,externalIds,citationCount,url"
LIMIT = 20

QUERIES = [
    "self-critique language model representations",
    "mechanistic interpretability residual stream",
    "self-reflection LLM internal states",
    "representation geometry transformer",
    "probing classifier transformer reasoning",
]

OUTPUT_PATH = "/workspaces/self-critique-drift-4703-claude/papers/semantic_scholar_results.json"


def search_with_retry(query: str, max_retries: int = 8) -> list[dict]:
    """Run a single search query with exponential backoff on 429."""
    params = {
        "query": query,
        "limit": LIMIT,
        "fields": FIELDS,
    }
    for attempt in range(max_retries):
        if attempt > 0:
            wait = min(30 * (2 ** (attempt - 1)), 120)
            print(f"  Retry {attempt}, waiting {wait}s ...", flush=True)
            time.sleep(wait)
        print(f"  GET {query!r} ... ", end="", flush=True)
        try:
            resp = requests.get(API_URL, params=params, timeout=30)
        except requests.exceptions.RequestException as e:
            print(f"connection error: {e}")
            continue
        if resp.status_code == 429:
            print(f"429 rate-limited")
            continue
        if resp.status_code != 200:
            print(f"HTTP {resp.status_code}")
            continue
        data = resp.json().get("data", [])
        print(f"{len(data)} results")
        return data
    print(f"  FAILED after {max_retries} retries")
    return []


def main():
    all_papers: dict[str, dict] = {}
    query_map: dict[str, list[str]] = {}

    for i, query in enumerate(QUERIES):
        print(f"\n[{i+1}/{len(QUERIES)}] Query: {query}")
        if i > 0:
            # Wait a generous 60s between queries to avoid rate limiting
            print(f"  Waiting 60s before next query...", flush=True)
            time.sleep(60)

        papers = search_with_retry(query)
        for paper in papers:
            pid = paper.get("paperId")
            if not pid:
                continue
            if pid not in all_papers:
                all_papers[pid] = paper
                query_map[pid] = []
            query_map[pid].append(query)

    # Attach which queries found each paper
    results = []
    for pid, paper in all_papers.items():
        paper["matchedQueries"] = query_map[pid]
        results.append(paper)

    # Sort by citation count descending
    results.sort(key=lambda p: p.get("citationCount") or 0, reverse=True)

    output = {
        "metadata": {
            "queries": QUERIES,
            "totalUnique": len(results),
            "fieldsRequested": FIELDS,
        },
        "papers": results,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved {len(results)} unique papers to {OUTPUT_PATH}")

    # Print summary
    print(f"\n--- Top 20 by citation count ---")
    for p in results[:20]:
        cites = p.get("citationCount") or 0
        year = p.get("year") or "n/a"
        title = p.get("title", "Untitled")
        print(f"  [{year}] ({cites:>5} cites) {title}")


if __name__ == "__main__":
    main()
