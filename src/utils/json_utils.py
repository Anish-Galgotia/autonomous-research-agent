import json
import re


def safe_json_loads(text: str) -> dict:
    """
    Safely parse JSON from LLM output.

    1. Try direct json.loads
    2. If that fails, extract JSON block using regex
    """

    # Attempt direct JSON parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: extract JSON object from text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError("No valid JSON found in LLM output")
