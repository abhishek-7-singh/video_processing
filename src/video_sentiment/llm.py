"""Unified LLM interface supporting Ollama with Gemini/OpenAI fallback."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Optional

from video_sentiment.errors import VideoSentimentError


def get_available_ollama_models(ollama_url: str = "http://127.0.0.1:11434") -> list[str]:
    """Retrieve list of locally pulled Ollama models."""
    req = urllib.request.Request(ollama_url.rstrip("/") + "/api/tags", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            return [str(m.get("name", "")) for m in payload.get("models", []) if m.get("name")]
    except Exception:
        return []


def resolve_model_name(preferred: str, ollama_url: str = "http://127.0.0.1:11434") -> str:
    """Resolve preferred model name against locally installed Ollama models."""
    available = get_available_ollama_models(ollama_url)
    if not available:
        return preferred
    # Exact match
    for m in available:
        if m == preferred or m.startswith(preferred + ":"):
            return m
    # Partial match
    for m in available:
        if preferred.split(":")[0] in m:
            return m
    # Fallback to first non-embedding model
    for m in available:
        if "embed" not in m.lower():
            return m
    return available[0]


def query_llm(
    prompt: str,
    system_instruction: str = "You are an expert video analyst and educational tutor.",
    ollama_url: str = "http://127.0.0.1:11434",
    ollama_model: str = "llama3.1",
    temperature: float = 0.2,
    timeout: float = 120,
    force_json: bool = False,
) -> str:
    """Query an LLM via Ollama, falling back to Gemini/OpenAI API if keys exist."""
    # 1. Check Gemini API key if present
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        try:
            return _query_gemini(prompt, system_instruction, gemini_key, force_json, timeout)
        except Exception as exc:
            print(f"[LLM] Gemini API call failed, falling back to Ollama: {exc}", flush=True)

    # 2. Check OpenAI API key if present
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        try:
            return _query_openai(prompt, system_instruction, openai_key, force_json, timeout)
        except Exception as exc:
            print(f"[LLM] OpenAI API call failed, falling back to Ollama: {exc}", flush=True)

    # 3. Use Ollama
    resolved_model = resolve_model_name(ollama_model, ollama_url)
    body: dict = {
        "model": resolved_model,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }
    if force_json:
        body["format"] = "json"

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        ollama_url.rstrip("/") + "/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            return str(payload.get("message", {}).get("content", ""))
    except Exception as exc:
        raise VideoSentimentError(
            f"Failed to query Ollama model '{resolved_model}' at {ollama_url}: {exc}. "
            f"Ensure Ollama is running and has a model (e.g. `ollama run qwen2.5-coder:3b`)."
        ) from exc


def _query_gemini(
    prompt: str, system: str, api_key: str, force_json: bool, timeout: float
) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    body: dict = {
        "contents": [{"parts": [{"text": prompt}]}],
        "systemInstruction": {"parts": [{"text": system}]},
    }
    if force_json:
        body["generationConfig"] = {"responseMimeType": "application/json"}
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        candidates = res.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                return str(parts[0].get("text", ""))
    return ""


def _query_openai(
    prompt: str, system: str, api_key: str, force_json: bool, timeout: float
) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    body: dict = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    if force_json:
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        choices = res.get("choices", [])
        if choices:
            return str(choices[0].get("message", {}).get("content", ""))
    return ""
