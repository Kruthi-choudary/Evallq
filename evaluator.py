from models import call_model, MODEL_REGISTRY
from metrics import judge_response
from cache import get_cached, set_cache
from database import EvaluationResult, Experiment
from sqlalchemy.orm import Session
from datetime import datetime


def evaluate_prompt(prompt: str, expected: str, model_keys: list, db: Session, experiment_id: int, api_keys: dict = None) -> list:
    api_keys = api_keys or {}
    results = []

    for model_key in model_keys:
        cached_response = get_cached(db, prompt, model_key)

        if cached_response:
            response_text = cached_response
            latency_ms = 0.0
            was_cached = True
        else:
            data = call_model(model_key, prompt, api_keys)
            response_text = data["response"]
            latency_ms = data["latency_ms"]
            was_cached = False
            set_cache(db, prompt, model_key, response_text)

        scores = judge_response(prompt, expected, response_text)
        reasoning = scores.pop("reasoning", "")
        overall = round(sum(scores.values()) / len(scores), 2)

        info = MODEL_REGISTRY.get(model_key, {})
        row = EvaluationResult(
            experiment_id=experiment_id,
            prompt=prompt,
            expected=expected,
            model_name=model_key,
            response=response_text,
            correctness=scores["correctness"],
            relevance=scores["relevance"],
            faithfulness=scores["faithfulness"],
            completeness=scores["completeness"],
            hallucination_free=scores["hallucination_free"],
            overall_score=overall,
            latency_ms=latency_ms,
            cached=was_cached,
            reasoning=reasoning,
            created_at=datetime.utcnow(),
        )
        db.add(row)
        db.commit()

        results.append({
            "model": model_key,
            "label": info.get("label", model_key),
            "company": info.get("company", ""),
            "response": response_text,
            "scores": scores,
            "overall": overall,
            "latency_ms": latency_ms,
            "cached": was_cached,
            "reasoning": reasoning,
        })

    return results
