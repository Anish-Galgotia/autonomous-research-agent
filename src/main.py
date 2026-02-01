import json

from src.agents.planner_agent import generate_research_plan
from src.agents.researcher_agent import execute_research
from src.agents.validator_agent import validate_research
from src.retriever.vector_store import retrieve_relevant_chunks
from src.generator.report_generator import generate_report
from src.utils.json_utils import safe_json_loads


CONFIDENCE_THRESHOLD = 0.7


def main():
    topic = "5G Core vendors comparison"

    print("[INFO] Starting autonomous research...\n")

    # 1. Planner
    plan = safe_json_loads(generate_research_plan(topic))

    # 2. First research pass
    vector_store = execute_research(plan["search_queries"])
    docs = retrieve_relevant_chunks(vector_store, topic, k=6)
    texts = [doc.page_content for doc in docs]

    validation = safe_json_loads(validate_research(texts))

    # 3. One controlled retry if confidence is low
    if validation["confidence_score"] < CONFIDENCE_THRESHOLD:
        print("[INFO] Initial confidence low. Running one refinement pass...\n")

        refined_queries = [
            "5G Core vendor architecture comparison",
            "5G Core security scalability challenges",
            "Ericsson Nokia Huawei 5G Core differences"
        ]

        vector_store = execute_research(refined_queries)
        docs = retrieve_relevant_chunks(vector_store, topic, k=6)
        texts = [doc.page_content for doc in docs]

        validation = safe_json_loads(validate_research(texts))

    print("[INFO] Research complete.")
    print(f"[INFO] Final confidence score: {validation['confidence_score']}\n")

    # 4. Final report generation
    print("[INFO] Generating final report...\n")
    final_report = generate_report(topic, texts)

    print("===== FINAL ANALYTICAL REPORT =====\n")
    print(final_report)


if __name__ == "__main__":
    main()

