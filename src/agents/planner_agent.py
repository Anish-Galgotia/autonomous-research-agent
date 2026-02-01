from langchain_openai import ChatOpenAI
from src.config.settings import MODEL_NAME, TEMPERATURE

def get_llm():
    return ChatOpenAI(
        model=MODEL_NAME,
        temperature=TEMPERATURE
    )

def generate_research_plan(topic: str) -> dict:
    """
    Generates a structured research plan for a given topic.
    """
    llm = get_llm()

    prompt = f"""
You are a senior technology research analyst.

Your task:
1. Break the topic into key research areas
2. Generate focused web search queries
3. Decide what information is critical vs optional

Topic:
{topic}

Respond ONLY in valid JSON with this format:

{{
  "research_areas": [
    "area 1",
    "area 2"
  ],
  "search_queries": [
    "query 1",
    "query 2"
  ],
  "critical_focus": [
    "must-know aspect 1",
    "must-know aspect 2"
  ]
}}
"""

    response = llm.invoke(prompt)

    return response.content
