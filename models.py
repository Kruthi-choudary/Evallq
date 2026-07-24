from groq import Groq
from dotenv import load_dotenv
import os
import time
import re

load_dotenv()

_groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# cost_in / cost_out = USD per 1M tokens
MODEL_REGISTRY = {
    "llama-3.3-70b": {"provider":"groq","model_id":"llama-3.3-70b-versatile","label":"LLaMA 3.3 70B","company":"Meta","free":True,"cost_in":0.0,"cost_out":0.0},
    "llama-3.1-8b":  {"provider":"groq","model_id":"llama-3.1-8b-instant","label":"LLaMA 3.1 8B","company":"Meta","free":True,"cost_in":0.0,"cost_out":0.0},
    "qwen-3.6-27b":  {"provider":"groq","model_id":"qwen/qwen3.6-27b","label":"Qwen 3.6 27B","company":"Alibaba","free":True,"cost_in":0.0,"cost_out":0.0},

    "gpt-4o":        {"provider":"openai","model_id":"gpt-4o","label":"GPT-4o","company":"OpenAI","free":False,"cost_in":2.50,"cost_out":10.0},
    "gpt-4o-mini":   {"provider":"openai","model_id":"gpt-4o-mini","label":"GPT-4o Mini","company":"OpenAI","free":False,"cost_in":0.15,"cost_out":0.60},
    "gpt-4.1":       {"provider":"openai","model_id":"gpt-4.1","label":"GPT-4.1","company":"OpenAI","free":False,"cost_in":2.00,"cost_out":8.0},
    "gpt-4.1-mini":  {"provider":"openai","model_id":"gpt-4.1-mini","label":"GPT-4.1 Mini","company":"OpenAI","free":False,"cost_in":0.40,"cost_out":1.60},
    "o1":            {"provider":"openai","model_id":"o1","label":"o1","company":"OpenAI","free":False,"cost_in":15.0,"cost_out":60.0},
    "o3-mini":       {"provider":"openai","model_id":"o3-mini","label":"o3 Mini","company":"OpenAI","free":False,"cost_in":1.10,"cost_out":4.40},

    "claude-opus":   {"provider":"anthropic","model_id":"claude-opus-4-8","label":"Claude Opus 4","company":"Anthropic","free":False,"cost_in":15.0,"cost_out":75.0},
    "claude-sonnet": {"provider":"anthropic","model_id":"claude-sonnet-4-6","label":"Claude Sonnet 4","company":"Anthropic","free":False,"cost_in":3.0,"cost_out":15.0},
    "claude-haiku":  {"provider":"anthropic","model_id":"claude-haiku-4-5-20251001","label":"Claude Haiku 4","company":"Anthropic","free":False,"cost_in":0.80,"cost_out":4.0},

    "gemini-2.5-pro":   {"provider":"gemini","model_id":"gemini-2.5-pro","label":"Gemini 2.5 Pro","company":"Google","free":False,"cost_in":1.25,"cost_out":10.0},
    "gemini-2.5-flash": {"provider":"gemini","model_id":"gemini-2.5-flash","label":"Gemini 2.5 Flash","company":"Google","free":False,"cost_in":0.30,"cost_out":2.50},
    "gemini-2.0-flash": {"provider":"gemini","model_id":"gemini-2.0-flash","label":"Gemini 2.0 Flash","company":"Google","free":False,"cost_in":0.10,"cost_out":0.40},
    "gemini-1.5-pro":   {"provider":"gemini","model_id":"gemini-1.5-pro","label":"Gemini 1.5 Pro","company":"Google","free":False,"cost_in":1.25,"cost_out":5.0},
}

AVAILABLE_MODELS = {k: v["model_id"] for k, v in MODEL_REGISTRY.items()}


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def call_model(model_key: str, prompt: str, api_keys: dict = None) -> dict:
    api_keys = api_keys or {}
    info = MODEL_REGISTRY.get(model_key)
    if not info:
        raise ValueError(f"Unknown model: {model_key}")

    provider = info["provider"]
    model_id = info["model_id"]
    start = time.time()
    input_tokens = output_tokens = 0

    if provider == "groq":
        response = _groq_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
        )
        text = _strip_thinking(response.choices[0].message.content.strip())
        input_tokens  = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=api_keys.get("openai"))
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content.strip()
        input_tokens  = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=api_keys.get("anthropic"))
        message = client.messages.create(
            model=model_id,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text = message.content[0].text.strip()
        input_tokens  = message.usage.input_tokens
        output_tokens = message.usage.output_tokens

    elif provider == "gemini":
        from google import genai
        client = genai.Client(api_key=api_keys.get("gemini"))
        response = client.models.generate_content(model=model_id, contents=prompt)
        text = response.text.strip()
        if response.usage_metadata:
            input_tokens  = response.usage_metadata.prompt_token_count or 0
            output_tokens = response.usage_metadata.candidates_token_count or 0

    else:
        raise ValueError(f"Unknown provider: {provider}")

    latency_ms = round((time.time() - start) * 1000, 2)
    cost_usd = round(
        (input_tokens / 1_000_000) * info["cost_in"] +
        (output_tokens / 1_000_000) * info["cost_out"], 6
    )
    return {
        "response": text,
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
    }
