# Evallq — LLM Evaluation Framework

A production-ready benchmarking platform that evaluates multiple AI models simultaneously across 5 quality metrics using **LLM-as-judge** scoring. Compare GPT, Claude, Gemini, and open-source models side by side — no human annotation needed.

**Live Demo:** https://evallq.onrender.com

---

## What it does

Run any prompt against up to 3 models at once. An independent LLaMA 3.3 70B judge automatically scores every response and builds a persistent leaderboard across all your evaluations.

## Features

- **16 models** — Meta, OpenAI, Anthropic, Google across free and paid tiers
- **5 eval metrics** — Correctness, Relevance, Faithfulness, Completeness, Hallucination-Free
- **LLM-as-judge** — automated scoring with reasoning explanation per response
- **Parallel model calls** — all selected models run concurrently via `ThreadPoolExecutor`
- **BYOK** — bring your own API keys for GPT, Claude, Gemini; stored in browser only, never on server
- **Groq models free** — LLaMA 3.3 70B, LLaMA 3.1 8B, Qwen 3.6 27B run with no key needed
- **Cost + token tracking** — per-call token counts and USD cost based on provider rates
- **Response caching** — SHA256-based cache avoids redundant API calls
- **Batch evaluation** — upload CSV/JSON datasets to evaluate at scale
- **Experiment tracking** — every run saved with full result history
- **Response comparison** — side-by-side model responses with scores per experiment prompt
- **Export CSV** — download full experiment results including scores, tokens, cost, and reasoning
- **Prompt library** — save and reload prompts locally (localStorage, no server needed)
- **Leaderboard** — ranked model comparison with bar chart and radar chart visualizations

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI (Python) |
| Database | PostgreSQL via Supabase |
| ORM | SQLAlchemy |
| LLM providers | Groq, OpenAI, Anthropic, Google Gemini |
| Frontend | Vanilla JS + Chart.js |
| Deployment | Render (backend) + Supabase (DB) |
| Containerization | Docker |

## Models Supported

**Free (Groq)**
- LLaMA 3.3 70B — Meta
- LLaMA 3.1 8B — Meta
- Qwen 3.6 27B — Alibaba

**BYOK — OpenAI**
- GPT-4o, GPT-4o Mini, GPT-4.1, GPT-4.1 Mini, o1, o3 Mini

**BYOK — Anthropic**
- Claude Opus 4, Claude Sonnet 4, Claude Haiku 4

**BYOK — Google**
- Gemini 2.5 Pro, Gemini 2.5 Flash, Gemini 2.0 Flash, Gemini 1.5 Pro

## How it works

```
User prompt → [Model A, Model B, Model C] → responses
                                                  ↓
                              LLaMA 3.3 70B judge scores each response
                                                  ↓
                         Scores saved to DB → Leaderboard updated
```

The judge uses a structured prompt to output JSON scores (0.0–1.0) for each metric. Overall score is the average across all 5 metrics.

## Running locally

```bash
git clone https://github.com/Kruthi-choudary/Evallq
cd Evallq

pip install -r requirements.txt

# create .env with your Groq key
echo "GROQ_API_KEY=your_key_here" > .env

uvicorn main:app --reload
```

Open `http://localhost:8000` — Groq models work immediately, add other keys via the API Keys button.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/models` | List all available models |
| POST | `/evaluate` | Run a single prompt evaluation |
| POST | `/batch` | Batch evaluate from CSV/JSON |
| GET | `/experiments` | List all experiment runs |
| GET | `/experiments/{id}` | Get full experiment results |
| GET | `/leaderboard` | Ranked model scores |

## Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Required — used for free models + judge |
| `DATABASE_URL` | PostgreSQL connection string (Supabase) |

User API keys (OpenAI, Anthropic, Google) are passed via request headers from the browser — never stored server-side.
