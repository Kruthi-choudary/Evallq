from groq import Groq
from dotenv import load_dotenv
import json
import os
import re

load_dotenv()

_judge_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
JUDGE_MODEL = "llama-3.3-70b-versatile"

_JUDGE_PROMPT = """You are a strict AI evaluation judge. Score the model response below on 5 metrics.

Question: {question}
Expected Answer: {expected}
Model Response: {response}

Score each from 0.0 to 1.0:
- correctness: Is the answer factually correct compared to the expected answer?
- relevance: Does the answer directly address what was asked?
- faithfulness: Is the answer grounded in facts with no fabricated details?
- completeness: Does the answer fully cover what was asked?
- hallucination_free: Does the answer avoid hallucinating or inventing information? (1.0 = no hallucination)

Return ONLY a JSON object, no explanation:
{{"correctness": 0.0, "relevance": 0.0, "faithfulness": 0.0, "completeness": 0.0, "hallucination_free": 0.0}}"""


def judge_response(question: str, expected: str, response: str) -> dict:
    prompt = _JUDGE_PROMPT.format(question=question, expected=expected or "N/A", response=response)

    try:
        result = _judge_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        text = result.choices[0].message.content.strip()
        match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
        if match:
            scores = json.loads(match.group())
            return {
                "correctness": round(float(scores.get("correctness", 0)), 2),
                "relevance": round(float(scores.get("relevance", 0)), 2),
                "faithfulness": round(float(scores.get("faithfulness", 0)), 2),
                "completeness": round(float(scores.get("completeness", 0)), 2),
                "hallucination_free": round(float(scores.get("hallucination_free", 0)), 2),
            }
    except Exception:
        pass

    return {
        "correctness": 0.0,
        "relevance": 0.0,
        "faithfulness": 0.0,
        "completeness": 0.0,
        "hallucination_free": 0.0,
    }
