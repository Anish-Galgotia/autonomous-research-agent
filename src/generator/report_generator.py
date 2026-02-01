from langchain_openai import ChatOpenAI
from src.config.settings import MODEL_NAME, TEMPERATURE


def generate_report(topic: str, context_chunks: list[str]) -> str:
    llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)

    context = "\n".join(context_chunks[:5])

    prompt = f"""
You are a technology consultant.

Using the context below, generate a concise analytical report on:

{topic}

Include:
- Executive summary
- Key vendors
- Pros & cons
- Key risks
- Final recommendation

Context:
{context}
"""

    response = llm.invoke(prompt)
    return response.content
