from src.ingestion.web_loader import load_web_content
from src.retriever.vector_store import build_vector_store


def generate_wiki_urls(query: str) -> list[str]:
    """
    Generate fallback Wikipedia URLs for a query.
    """
    base = "https://en.wikipedia.org/wiki/"
    words = query.replace(" ", "_")

    return [
        base + words,
        base + query.split()[0],              # e.g. "5G"
        base + "5G",                           # ultimate fallback
        base + "Telecommunications"
    ]


def execute_research(search_queries: list[str]):
    collected_texts = []

    for query in search_queries:
        print(f"[INFO] Researching query: {query}")

        urls = generate_wiki_urls(query)

        for url in urls:
            text = load_web_content(url)
            if text:
                collected_texts.append(text)
                print(f"[INFO] Collected data from: {url}")
                break   # stop after first success

    if not collected_texts:
        raise ValueError("No research content could be collected.")

    vector_store = build_vector_store(collected_texts)
    return vector_store
