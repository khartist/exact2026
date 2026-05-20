"""Plain text formula/law search for Type 2 planning."""

from __future__ import annotations

import json
import re
from typing import Any

from .formula_bank import FORMULAS, LAW_STATEMENTS
from .schemas import Formula, LawStatement


def search_physics_knowledge(query: str, top_k: int = 8) -> list[dict[str, Any]]:
    query_tokens = tokenize(query)
    if not query_tokens:
        return []
    scored: list[tuple[int, int, dict[str, Any]]] = []
    for index, item in enumerate([*FORMULAS, *LAW_STATEMENTS]):
        payload = knowledge_to_payload(item)
        item_tokens = tokenize(" ".join(searchable_fields(payload)))
        overlap = query_tokens.intersection(item_tokens)
        if not overlap:
            continue
        score = len(overlap)
        keyword_hits = query_tokens.intersection(tokenize(" ".join(payload.get("keywords", []))))
        score += 2 * len(keyword_hits)
        scored.append((score, -index, payload))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [payload for _, _, payload in scored[: max(0, top_k)]]


def formula_search_tool(query: str, top_k: int = 8) -> str:
    return json.dumps(search_physics_knowledge(query, top_k), ensure_ascii=False)


def tokenize(text: str) -> set[str]:
    raw = re.findall(r"[A-Za-z0-9_]+", text.lower().replace("-", " "))
    tokens = set(raw)
    expanded = set(tokens)
    for token in tokens:
        if token.endswith("s") and len(token) > 3:
            expanded.add(token[:-1])
    return expanded


def searchable_fields(payload: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in (
        "id",
        "topic",
        "equation",
        "expression",
        "description",
        "statement",
    ):
        value = payload.get(key)
        if value:
            values.append(str(value))
    for key in ("keywords", "aliases", "related_formula_ids"):
        value = payload.get(key)
        if isinstance(value, (list, tuple)):
            values.extend(str(item) for item in value)
    return values


def knowledge_to_payload(item: Formula | LawStatement) -> dict[str, Any]:
    if isinstance(item, Formula):
        return {
            "type": "formula",
            "id": item.id,
            "topic": item.topic,
            "equation": item.equation,
            "expression": item.expression,
            "intermediates": list(item.intermediates),
            "required_variables": list(item.required_variables),
            "target_variables": list(item.target_variables),
            "output_unit": item.output_unit,
            "keywords": list(item.keywords),
            "description": item.description,
        }
    return {
        "type": "law",
        "id": item.id,
        "topic": item.topic,
        "statement": item.statement,
        "keywords": list(item.keywords),
        "aliases": list(item.aliases),
        "related_formula_ids": list(item.related_formula_ids),
    }
