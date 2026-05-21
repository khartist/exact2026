"""Plain text formula/law search and modular planner context for Type 2 planning."""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Literal

from .formula_bank import (
    CATEGORIES,
    CATEGORY_INDEX,
    CONSTANT_INDEX,
    CONSTANTS,
    DIAGRAM_INDEX,
    DIAGRAMS,
    EXAMPLE_INDEX,
    FORMULA_INDEX,
    FORMULAS,
    LAW_INDEX,
    LAW_STATEMENTS,
    SYMBOL_INDEX,
    SYMBOLS,
    WORKED_EXAMPLES,
)
from .schemas import (
    Category,
    Diagram,
    Formula,
    KnowledgeSearchConfig,
    LawStatement,
    PhysicalConstant,
    SymbolDefinition,
    WorkedExample,
)

ContextType = Literal["formulas", "laws", "symbols", "constants", "examples", "categories", "diagrams"]


def search_physics_knowledge(
    query: str,
    top_k: int = 8,
    config: KnowledgeSearchConfig | None = None,
) -> list[dict[str, Any]]:
    """Return a backwards-compatible flat top-k list of formula/law payloads."""

    active_config = config or KnowledgeSearchConfig(top_k=top_k, enabled_types=("formulas", "laws"))
    primary = score_items(query, active_config, ("formulas", "laws"))
    return [payload for _, _, payload, _ in primary[: max(0, active_config.top_k)]]


def build_planner_knowledge_context(
    query: str,
    config: KnowledgeSearchConfig | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Return grouped, configurable context for the planner prompt."""

    active_config = config or KnowledgeSearchConfig()
    enabled = set(active_config.enabled_types)
    context: dict[str, list[dict[str, Any]]] = {
        "formulas": [],
        "laws": [],
        "symbols": [],
        "constants": [],
        "examples": [],
        "categories": [],
        "diagrams": [],
    }
    if not tokenize(query):
        return {key: value for key, value in context.items() if key in enabled}

    primary = score_items(query, active_config, enabled)
    selected_formula_ids: list[str] = []
    selected_law_ids: list[str] = []
    selected_symbol_ids: list[str] = []
    selected_constant_ids: list[str] = []
    selected_example_ids: list[str] = []
    selected_category_ids: list[str] = []
    selected_diagram_ids: list[str] = []

    for _, _, payload, item in primary:
        item_type = payload["type"]
        if item_type == "formula" and "formulas" in enabled:
            append_unique(selected_formula_ids, payload["id"])
        elif item_type == "law" and "laws" in enabled:
            append_unique(selected_law_ids, payload["id"])
        elif item_type == "symbol" and "symbols" in enabled:
            append_unique(selected_symbol_ids, payload["symbol"])
        elif item_type == "constant" and "constants" in enabled:
            append_unique(selected_constant_ids, payload["symbol"])
        elif item_type == "example" and "examples" in enabled:
            append_unique(selected_example_ids, payload["id"])
        elif item_type == "category" and "categories" in enabled:
            append_unique(selected_category_ids, payload["id"])
        elif item_type == "diagram" and "diagrams" in enabled:
            append_unique(selected_diagram_ids, payload["id"])

        if isinstance(item, Formula):
            collect_formula_refs(item, selected_symbol_ids, selected_constant_ids, enabled)
        elif isinstance(item, LawStatement):
            collect_law_refs(item, selected_formula_ids, selected_diagram_ids, enabled)
        elif isinstance(item, WorkedExample):
            collect_example_refs(item, selected_formula_ids, selected_law_ids, enabled)
        elif isinstance(item, Category):
            collect_category_refs(item, selected_formula_ids, selected_law_ids, selected_example_ids, selected_symbol_ids, selected_constant_ids, selected_diagram_ids, enabled)
        elif isinstance(item, Diagram):
            collect_diagram_refs(item, selected_formula_ids, selected_law_ids, enabled)

    selected_formula_ids = selected_formula_ids[: active_config.top_k]
    selected_law_ids = selected_law_ids[: active_config.top_k]
    selected_example_ids = selected_example_ids[: active_config.max_examples]
    selected_diagram_ids = selected_diagram_ids[: active_config.max_diagrams]

    if "formulas" in enabled:
        context["formulas"] = [knowledge_to_payload(FORMULA_INDEX[item_id]) for item_id in selected_formula_ids if item_id in FORMULA_INDEX]
    if "laws" in enabled:
        context["laws"] = [knowledge_to_payload(LAW_INDEX[item_id]) for item_id in selected_law_ids if item_id in LAW_INDEX]
    if "symbols" in enabled:
        context["symbols"] = [knowledge_to_payload(SYMBOL_INDEX[item_id]) for item_id in selected_symbol_ids if item_id in SYMBOL_INDEX]
    if "constants" in enabled:
        context["constants"] = [knowledge_to_payload(CONSTANT_INDEX[item_id]) for item_id in selected_constant_ids if item_id in CONSTANT_INDEX]
    if "examples" in enabled:
        context["examples"] = [knowledge_to_payload(EXAMPLE_INDEX[item_id]) for item_id in selected_example_ids if item_id in EXAMPLE_INDEX]
    if "categories" in enabled:
        context["categories"] = [knowledge_to_payload(CATEGORY_INDEX[item_id]) for item_id in selected_category_ids if item_id in CATEGORY_INDEX]
    if "diagrams" in enabled:
        context["diagrams"] = [knowledge_to_payload(DIAGRAM_INDEX[item_id]) for item_id in selected_diagram_ids if item_id in DIAGRAM_INDEX]

    return {key: value for key, value in context.items() if key in enabled}


def formula_search_tool(query: str, top_k: int = 8) -> str:
    return json.dumps(search_physics_knowledge(query, top_k), ensure_ascii=False)


def score_items(
    query: str,
    config: KnowledgeSearchConfig,
    enabled_types: Iterable[str],
) -> list[tuple[int, int, dict[str, Any], object]]:
    query_tokens = tokenize(query)
    if not query_tokens:
        return []
    enabled = set(enabled_types)
    candidates: list[object] = []
    if "formulas" in enabled:
        candidates.extend(FORMULAS)
    if "laws" in enabled:
        candidates.extend(LAW_STATEMENTS)
    if "symbols" in enabled:
        candidates.extend(SYMBOLS)
    if "constants" in enabled:
        candidates.extend(CONSTANTS)
    if "examples" in enabled:
        candidates.extend(WORKED_EXAMPLES)
    if "categories" in enabled:
        candidates.extend(CATEGORIES)
    if "diagrams" in enabled:
        candidates.extend(DIAGRAMS)

    scored: list[tuple[int, int, dict[str, Any], object]] = []
    for index, item in enumerate(candidates):
        payload = knowledge_to_payload(item)
        item_tokens = tokenize(" ".join(searchable_fields(payload)))
        overlap = query_tokens.intersection(item_tokens)
        if not overlap:
            continue
        score = len(overlap)
        keyword_hits = query_tokens.intersection(tokenize(" ".join(payload.get("keywords", []))))
        score += 2 * len(keyword_hits)
        scored.append((score, -index, payload, item))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return scored[: max(config.top_k * 3, config.top_k)]


def collect_formula_refs(
    formula: Formula,
    symbol_ids: list[str],
    constant_ids: list[str],
    enabled: set[str],
) -> None:
    if "symbols" in enabled:
        for name in (*formula.required_variables, *formula.target_variables):
            append_unique(symbol_ids, name)
    if "constants" in enabled:
        for name in formula.required_variables:
            if name in {"k", "eps0", "e", "ln2"}:
                append_unique(constant_ids, name)


def collect_law_refs(
    law: LawStatement,
    formula_ids: list[str],
    diagram_ids: list[str],
    enabled: set[str],
) -> None:
    if "formulas" in enabled:
        for formula_id in law.related_formula_ids:
            append_unique(formula_ids, formula_id)
    if "diagrams" in enabled:
        for diagram_id in law.diagram_ids:
            append_unique(diagram_ids, diagram_id)


def collect_example_refs(
    example: WorkedExample,
    formula_ids: list[str],
    law_ids: list[str],
    enabled: set[str],
) -> None:
    if "formulas" in enabled:
        for formula_id in example.related_formula_ids:
            append_unique(formula_ids, formula_id)
    if "laws" in enabled:
        for law_id in example.related_law_ids:
            append_unique(law_ids, law_id)


def collect_category_refs(
    category: Category,
    formula_ids: list[str],
    law_ids: list[str],
    example_ids: list[str],
    symbol_ids: list[str],
    constant_ids: list[str],
    diagram_ids: list[str],
    enabled: set[str],
) -> None:
    if "formulas" in enabled:
        for formula_id in category.formula_ids:
            append_unique(formula_ids, formula_id)
    if "laws" in enabled:
        for law_id in category.law_ids:
            append_unique(law_ids, law_id)
    if "examples" in enabled:
        for example_id in category.example_ids:
            append_unique(example_ids, example_id)
    if "symbols" in enabled:
        for symbol_id in category.symbol_ids:
            append_unique(symbol_ids, symbol_id)
    if "constants" in enabled:
        for constant_id in category.constant_ids:
            append_unique(constant_ids, constant_id)
    if "diagrams" in enabled:
        for diagram_id in category.diagram_ids:
            append_unique(diagram_ids, diagram_id)


def collect_diagram_refs(
    diagram: Diagram,
    formula_ids: list[str],
    law_ids: list[str],
    enabled: set[str],
) -> None:
    if "formulas" in enabled:
        for formula_id in diagram.related_formula_ids:
            append_unique(formula_ids, formula_id)
    if "laws" in enabled:
        for law_id in diagram.related_law_ids:
            append_unique(law_ids, law_id)


def append_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


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
        "symbol",
        "topic",
        "category",
        "title",
        "equation",
        "expression",
        "description",
        "statement",
        "meaning",
        "summary",
        "prompt",
    ):
        value = payload.get(key)
        if value:
            values.append(str(value))
    for key in (
        "keywords",
        "aliases",
        "related_formula_ids",
        "related_law_ids",
        "givens",
        "steps",
        "answer",
        "formula_ids",
        "law_ids",
        "example_ids",
        "symbol_ids",
        "constant_ids",
    ):
        value = payload.get(key)
        if isinstance(value, (list, tuple)):
            values.extend(str(item) for item in value)
    return values


def knowledge_to_payload(item: object) -> dict[str, Any]:
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
            "category": item.category,
            "assumptions": list(item.assumptions),
            "pitfalls": list(item.pitfalls),
        }
    if isinstance(item, LawStatement):
        return {
            "type": "law",
            "id": item.id,
            "topic": item.topic,
            "statement": item.statement,
            "keywords": list(item.keywords),
            "aliases": list(item.aliases),
            "related_formula_ids": list(item.related_formula_ids),
            "category": item.category,
            "diagram_ids": list(item.diagram_ids),
        }
    if isinstance(item, SymbolDefinition):
        return {
            "type": "symbol",
            "symbol": item.symbol,
            "meaning": item.meaning,
            "si_unit": item.si_unit,
            "aliases": list(item.aliases),
            "notes": item.notes,
        }
    if isinstance(item, PhysicalConstant):
        return {
            "type": "constant",
            "symbol": item.symbol,
            "value": item.value,
            "si_unit": item.si_unit,
            "description": item.description,
            "aliases": list(item.aliases),
        }
    if isinstance(item, WorkedExample):
        return {
            "type": "example",
            "id": item.id,
            "category": item.category,
            "title": item.title,
            "prompt": item.prompt,
            "givens": list(item.givens),
            "steps": list(item.steps),
            "answer": list(item.answer),
            "related_formula_ids": list(item.related_formula_ids),
            "related_law_ids": list(item.related_law_ids),
        }
    if isinstance(item, Category):
        return {
            "type": "category",
            "id": item.id,
            "title": item.title,
            "summary": item.summary,
            "formula_ids": list(item.formula_ids),
            "law_ids": list(item.law_ids),
            "example_ids": list(item.example_ids),
            "symbol_ids": list(item.symbol_ids),
            "constant_ids": list(item.constant_ids),
            "diagram_ids": list(item.diagram_ids),
        }
    if isinstance(item, Diagram):
        return {
            "type": "diagram",
            "id": item.id,
            "title": item.title,
            "mermaid": item.mermaid,
            "related_formula_ids": list(item.related_formula_ids),
            "related_law_ids": list(item.related_law_ids),
        }
    raise TypeError(f"Unsupported knowledge item: {type(item).__name__}")
