from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, Form, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
import csv
import io
import json
from datetime import datetime

from database import get_db, init_db, Experiment, EvaluationResult
from evaluator import evaluate_prompt
from models import MODEL_REGISTRY

app = FastAPI(title="LLM Evaluation Framework")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

init_db()


def _extract_keys(request: Request) -> dict:
    return {
        "openai": request.headers.get("X-OpenAI-Key"),
        "anthropic": request.headers.get("X-Anthropic-Key"),
        "gemini": request.headers.get("X-Gemini-Key"),
    }


class EvaluateRequest(BaseModel):
    prompt: str
    expected: Optional[str] = ""
    models: Optional[List[str]] = None
    experiment_name: Optional[str] = "Single Evaluation"


@app.get("/")
def root():
    return FileResponse("index.html")


@app.get("/models")
def list_models():
    return [
        {
            "key": k,
            "label": v["label"],
            "company": v["company"],
            "provider": v["provider"],
            "free": v["free"],
        }
        for k, v in MODEL_REGISTRY.items()
    ]


@app.post("/evaluate")
def evaluate(req: EvaluateRequest, request: Request, db: Session = Depends(get_db)):
    api_keys = _extract_keys(request)

    free_models = [k for k, v in MODEL_REGISTRY.items() if v["free"]]
    models_to_test = req.models or free_models

    # validate keys for non-free models
    for model_key in models_to_test:
        info = MODEL_REGISTRY.get(model_key, {})
        if not info.get("free"):
            provider = info.get("provider")
            if not api_keys.get(provider):
                raise HTTPException(400, f"API key required for {info.get('label')} — add your {provider.title()} key in Settings")

    experiment = Experiment(
        name=req.experiment_name,
        created_at=datetime.utcnow(),
        total_prompts=1,
        models_tested=models_to_test,
    )
    db.add(experiment)
    db.commit()
    db.refresh(experiment)

    results = evaluate_prompt(req.prompt, req.expected, models_to_test, db, experiment.id, api_keys)
    return {"experiment_id": experiment.id, "prompt": req.prompt, "expected": req.expected, "results": results}


@app.post("/batch")
async def batch_evaluate(
    request: Request,
    file: UploadFile = File(...),
    experiment_name: str = Form(default="Batch Evaluation"),
    models: str = Form(default=""),
    db: Session = Depends(get_db),
):
    api_keys = _extract_keys(request)
    content = await file.read()

    if file.filename.endswith(".csv"):
        reader = csv.DictReader(io.StringIO(content.decode()))
        rows = list(reader)
    elif file.filename.endswith(".json"):
        rows = json.loads(content)
    else:
        raise HTTPException(400, "Only CSV or JSON files are supported")

    if not rows:
        raise HTTPException(400, "File is empty")

    free_models = [k for k, v in MODEL_REGISTRY.items() if v["free"]]
    models_to_test = [m.strip() for m in models.split(",") if m.strip()] or free_models

    experiment = Experiment(
        name=experiment_name,
        created_at=datetime.utcnow(),
        total_prompts=len(rows),
        models_tested=models_to_test,
    )
    db.add(experiment)
    db.commit()
    db.refresh(experiment)

    all_results = []
    for row in rows:
        prompt = row.get("prompt") or row.get("question", "")
        expected = row.get("expected") or row.get("answer", "")
        if not prompt:
            continue
        results = evaluate_prompt(prompt, expected, models_to_test, db, experiment.id, api_keys)
        all_results.append({"prompt": prompt, "expected": expected, "results": results})

    return {"experiment_id": experiment.id, "total": len(all_results), "results": all_results}


@app.get("/experiments")
def list_experiments(db: Session = Depends(get_db)):
    experiments = db.query(Experiment).order_by(Experiment.created_at.desc()).all()
    output = []
    for exp in experiments:
        rows = db.query(EvaluationResult).filter(EvaluationResult.experiment_id == exp.id).all()
        avg = round(sum(r.overall_score for r in rows) / len(rows), 2) if rows else 0
        output.append({
            "id": exp.id,
            "name": exp.name,
            "created_at": exp.created_at.isoformat(),
            "total_prompts": exp.total_prompts,
            "models_tested": exp.models_tested,
            "avg_score": avg,
        })
    return output


@app.get("/experiments/{experiment_id}")
def get_experiment(experiment_id: int, db: Session = Depends(get_db)):
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(404, "Experiment not found")

    rows = db.query(EvaluationResult).filter(EvaluationResult.experiment_id == experiment_id).all()
    return {
        "id": experiment.id,
        "name": experiment.name,
        "created_at": experiment.created_at.isoformat(),
        "models_tested": experiment.models_tested,
        "results": [
            {
                "prompt": r.prompt,
                "expected": r.expected,
                "model_name": r.model_name,
                "response": r.response,
                "scores": {
                    "correctness": r.correctness,
                    "relevance": r.relevance,
                    "faithfulness": r.faithfulness,
                    "completeness": r.completeness,
                    "hallucination_free": r.hallucination_free,
                },
                "overall_score": r.overall_score,
                "latency_ms": r.latency_ms,
                "cached": r.cached,
                "reasoning": r.reasoning or "",
            }
            for r in rows
        ],
    }


@app.get("/leaderboard")
def leaderboard(db: Session = Depends(get_db)):
    model_names = [m[0] for m in db.query(EvaluationResult.model_name).distinct().all()]
    result = []

    for model in model_names:
        rows = db.query(EvaluationResult).filter(EvaluationResult.model_name == model).all()
        if not rows:
            continue

        total = len(rows)
        live_rows = [r for r in rows if not r.cached]
        avg_latency = round(sum(r.latency_ms for r in live_rows) / len(live_rows), 1) if live_rows else 0
        info = MODEL_REGISTRY.get(model, {})

        result.append({
            "model": model,
            "label": info.get("label", model),
            "company": info.get("company", ""),
            "correctness": round(sum(r.correctness for r in rows) / total, 2),
            "relevance": round(sum(r.relevance for r in rows) / total, 2),
            "faithfulness": round(sum(r.faithfulness for r in rows) / total, 2),
            "completeness": round(sum(r.completeness for r in rows) / total, 2),
            "hallucination_free": round(sum(r.hallucination_free for r in rows) / total, 2),
            "overall": round(sum(r.overall_score for r in rows) / total, 2),
            "avg_latency_ms": avg_latency,
            "total_evals": total,
            "cache_hits": sum(1 for r in rows if r.cached),
            "total_cost_usd": round(sum(r.cost_usd or 0 for r in rows), 4),
            "avg_cost_usd": round(sum(r.cost_usd or 0 for r in live_rows) / len(live_rows), 6) if live_rows else 0,
        })

    result.sort(key=lambda x: x["overall"], reverse=True)
    for i, item in enumerate(result):
        item["rank"] = i + 1

    return result
