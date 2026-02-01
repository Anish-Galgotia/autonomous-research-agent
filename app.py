import streamlit as st

from src.agents.planner_agent import generate_research_plan
from src.agents.researcher_agent import execute_research
from src.agents.validator_agent import validate_research
from src.retriever.vector_store import retrieve_relevant_chunks
from src.generator.report_generator import generate_report
from src.utils.json_utils import safe_json_loads


CONFIDENCE_THRESHOLD = 0.7


st.set_page_config(page_title="Autonomous Research Agent", layout="wide")

st.title("Autonomous Research Analyst Agent")
st.write(
    "This AI agent autonomously plans research, ingests real-world data, "
    "self-evaluates confidence, and generates a consultant-style report."
)

topic = st.text_input(
    "Enter a research topic",
    placeholder="e.g. 5G Core vendors comparison"
)

run_button = st.button("Run Research")

if run_button and topic:
    with st.spinner("🔍 Planning and researching..."):
        # Planner
        plan = safe_json_loads(generate_research_plan(topic))

        # First research pass
        vector_store = execute_research(plan["search_queries"])
        docs = retrieve_relevant_chunks(vector_store, topic, k=6)
        texts = [doc.page_content for doc in docs]

        validation = safe_json_loads(validate_research(texts))

        # One controlled retry
        if validation["confidence_score"] < CONFIDENCE_THRESHOLD:
            refined_queries = [
                f"{topic} architecture comparison",
                f"{topic} security and scalability challenges",
                f"major vendors involved in {topic}"
            ]

            vector_store = execute_research(refined_queries)
            docs = retrieve_relevant_chunks(vector_store, topic, k=6)
            texts = [doc.page_content for doc in docs]

            validation = safe_json_loads(validate_research(texts))

    st.success(
        f"Research completed | Confidence score: {validation['confidence_score']}"
    )

    st.subheader("Final Analytical Report")
    report = generate_report(topic, texts)
    st.write(report)
