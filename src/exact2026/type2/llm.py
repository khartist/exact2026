"""LangChain model construction for Type 2 agents."""

from __future__ import annotations

from typing import Any

from .schemas import Type2SolverConfig


def build_type2_llm(config: Type2SolverConfig) -> Any:
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise RuntimeError(
            "Type 2 LangChain agents require langchain-ollama. Run `uv sync`."
        ) from exc

    return ChatOllama(
        model=config.model,
        base_url=normalize_ollama_base_url(config.ollama_url),
        temperature=config.temperature,
        timeout=config.timeout,
    )


def normalize_ollama_base_url(url: str) -> str:
    return url.removesuffix("/api/chat").rstrip("/")


def message_content(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "\n".join(str(item) for item in content)
    return str(content)
