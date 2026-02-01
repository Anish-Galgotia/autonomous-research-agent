from langchain_openai import ChatOpenAI
from src.config.settings import MODEL_NAME, TEMPERATURE, CONFIDENCE_THRESHOLD


def validate_research(retrieved_chunks: list[str]) -> dict:
    """
    Evaluates whether collected research is sufficient.
    """
    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=TEMPERATURE
    )

    content_preview = "\n".join(retrieved_chunks[:5])

    prompt = f"""
You are a senior research quality auditor.

Given the following research excerpts, evaluate:
1. Coverage breadth
2. Technical depth
3. Redundancy
4. Missing critical aspects

Provide:
- A confidence score between 0 and 1
- A short justification
- Whether more research is required

Research excerpts:
{content_preview}

Respond ONLY in valid JSON:

{{
  "confidence_score": 0.0,
  "justification": "string",
  "needs_more_research": true
}}
"""

    response = llm.invoke(prompt)
    return response.content
