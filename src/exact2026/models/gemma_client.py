"""Ollama chat client used by the MVP pipeline."""

from __future__ import annotations

import json
from typing import Any

import requests


DEFAULT_MODEL = "gemma4:e2b-it-q4_K_M"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"


class OllamaClient:
    def __init__(
        self,
        url: str = DEFAULT_OLLAMA_URL,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        timeout: float = 180.0,
    ) -> None:
        self.url = url
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {"temperature": self.temperature},
        }
        response = requests.post(self.url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        body = response.json()
        try:
            return str(body["message"]["content"])
        except KeyError as exc:
            raise RuntimeError(f"Unexpected Ollama response: {json.dumps(body)}") from exc

