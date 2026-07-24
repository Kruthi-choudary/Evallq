from concurrent.futures import ThreadPoolExecutor
from models import call_model, MODEL_REGISTRY
from metrics import judge_response
from cache import get_cached, set_cache
from database import EvaluationResult
from sqlalchemy.orm import Session
from datetime import datetime


def evaluate_prompt(prompt: str, expected: str, model_keys: list, db: Session, experiment_id: int, api_keys: dict = None) -> list:
    api_keys = api_keys or {}

    # Phase 1 — cache check (sequential, uses DB)
    cache_map = {k: get_cached(db, prompt, k) for k in model_keys}

    # Phase 2 — model calls + judging (parallel, no DB)
    def process(model_key):
        cached = cache_map[model_key]
        if cached:
            response_text, latency_ms, was_cached = cached, 0.0, True
            input_tokens = output_tokens = cost_usd = 0
        else:
            data = call_model(model_key, prompt, api_keys)
            response_text, latency_ms, was_cached = data["response"], data["latency_ms"], False
            input_tokens  = data.get("input_tokens", 0)
            output_tokens = data.get("output_tokens", 0)
            cost_usd      = data.get("cost_usd", 0.0)

        scores = judge_response(prompt, expected, response_text)
        reasoning = scores.pop("reasoning", "")
        overall = round(sum(scores.values()) / len(scores), 2)
        return model_key, response_text, latency_ms, was_cached, scores, reasoning, overall, input_tokens, output_tokens, cost_usd

    with ThreadPoolExecutor(max_workers=min(len(model_keys), 5)) as ex:
        processed = list(ex.map(process, model_keys))

    # Phase 3 — save to DB + build response (sequential, uses DB)
    results = []
    for model_key, response_text, latency_ms, was_cached, scores, reasoning, overall, input_tokens, output_tokens, cost_usd in processed:
        if not was_cached:
            set_cache(db, prompt, model_key, response_text)

        info = MODEL_REGISTRY.get(model_key, {})
        db.add(EvaluationResult(
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
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            created_at=datetime.utcnow(),
        ))
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
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
        })

    return results
