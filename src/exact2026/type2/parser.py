"""Parser for Type 2 physics questions."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from .schemas import KnownValue, ParsedProblem
from .units import convert_value


ModelCompleteFn = Callable[[str, str], str]

PARSER_SYSTEM_PROMPT = """You extract structured variables from physics problems.
Return strict JSON only with keys: topic, knowns, target, target_unit, possible_formulas.
knowns must map symbols to objects with value and unit. Do not solve arithmetic."""


SYMBOL_WORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("X_L", ("inductive reactance", "inductor reactance")),
    ("X_C", ("capacitive reactance", "capacitor reactance")),
    ("Z", ("impedance",)),
    ("P", ("power", "dissipated")),
    ("I", ("current",)),
    ("V", ("voltage", "potential difference", "potential")),
    ("R", ("resistance", "resistor")),
    ("L", ("inductance", "inductor")),
    ("C", ("capacitance", "capacitor")),
    ("E", ("electric field", "energy")),
    ("F", ("force",)),
)

UNIT_PATTERN = (
    r"(?:k?Ω|k?Ω|ohms?|m?V|kV|volts?|m?A|amps?|watts?|W|Hz|hertz|m?H|henr(?:y|ies)|"
    r"[unpμµ]?F|farads?|Hz|hertz|centimeters?|cm|n?C|uC|μC|µC|coulombs?|N/C|N|meters?|m|J)"
)


def parse_question(question: str, model_complete: ModelCompleteFn | None = None) -> ParsedProblem:
    llm_data = parse_with_llm(question, model_complete) if model_complete else {}
    knowns = parse_knowns_from_llm(llm_data)
    knowns.update(extract_knowns_regex(question))

    target = clean_symbol(str(llm_data.get("target", ""))) or infer_target(question, knowns)
    target_unit = str(llm_data.get("target_unit", "")).strip() or infer_target_unit(target)
    topic = str(llm_data.get("topic", "")).strip() or infer_topic(question, target)
    possible = llm_data.get("possible_formulas", [])
    possible_formulas = [str(item) for item in possible] if isinstance(possible, list) else []

    return ParsedProblem(
        question=question,
        topic=topic,
        knowns=knowns,
        target=target,
        target_unit=target_unit,
        possible_formulas=possible_formulas,
    )


def parse_with_llm(question: str, model_complete: ModelCompleteFn | None) -> dict[str, Any]:
    if model_complete is None:
        return {}
    user_prompt = "\n".join(
        [
            "Physics question:",
            question,
            "",
            "Return strict JSON only.",
        ]
    )
    try:
        raw = model_complete(PARSER_SYSTEM_PROMPT, user_prompt)
    except Exception:
        return {}
    parsed = extract_json_object(raw)
    return parsed if isinstance(parsed, dict) else {}


def parse_knowns_from_llm(data: dict[str, Any]) -> dict[str, KnownValue]:
    raw_knowns = data.get("knowns")
    if not isinstance(raw_knowns, dict):
        return {}
    knowns: dict[str, KnownValue] = {}
    for raw_symbol, raw_value in raw_knowns.items():
        symbol = normalize_input_symbol(str(raw_symbol))
        if not symbol or not isinstance(raw_value, dict):
            continue
        try:
            value = float(raw_value.get("value"))
        except (TypeError, ValueError):
            continue
        unit = str(raw_value.get("unit", "")).strip()
        si_value, si_unit = convert_value(value, unit)
        knowns[symbol] = KnownValue(symbol, value, unit, si_value, si_unit, f"{value} {unit}".strip())
    return knowns


def extract_knowns_regex(question: str) -> dict[str, KnownValue]:
    knowns: dict[str, KnownValue] = {}
    normalized = normalize_text(question)
    add_explicit_symbol_matches(normalized, knowns)
    add_distance_label_matches(normalized, knowns)
    add_word_matches(normalized, knowns)
    add_constant_k(normalized, knowns)
    return knowns


def add_explicit_symbol_matches(text: str, knowns: dict[str, KnownValue]) -> None:
    pattern = re.compile(
        rf"\b(?P<symbol>R\d*|V|U|I|P|L|C|f|q\d*|Q|r|E|F|k|Z|X_L|XL|X_C|XC)\b"
        rf"\s*(?:=|is|of)?\s*(?P<value>{NUMBER_PATTERN})\s*"
        rf"(?P<unit>{UNIT_PATTERN})?",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        symbol = normalize_input_symbol(match.group("symbol"))
        unit = match.group("unit") or infer_unit_from_symbol(symbol)
        add_known(knowns, symbol, match.group("value"), unit)


def add_word_matches(text: str, knowns: dict[str, KnownValue]) -> None:
    for symbol, words in SYMBOL_WORDS:
        for word in words:
            pattern = re.compile(
                rf"\b{re.escape(word)}\b(?:\s+(?:is|of|equals?))?\s*"
                rf"(?P<value>{NUMBER_PATTERN})\s*(?P<unit>{UNIT_PATTERN})",
                re.IGNORECASE,
            )
            for match in pattern.finditer(text):
                add_known(knowns, symbol, match.group("value"), match.group("unit"))


NUMBER_PATTERN = r"[-+]?\d+(?:\.\d+)?(?:\s*(?:×|x|\*)\s*10\s*\^?\s*[-+]?\d+|e[-+]?\d+)?"


def add_distance_label_matches(text: str, knowns: dict[str, KnownValue]) -> None:
    pattern = re.compile(
        rf"\b(?P<label>AB|AC|BC|r\d*)\b\s*(?:=|is|of)?\s*"
        rf"(?P<value>{NUMBER_PATTERN})\s*(?P<unit>{UNIT_PATTERN})",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        add_known(
            knowns,
            match.group("label").upper(),
            match.group("value"),
            match.group("unit"),
        )


def add_constant_k(text: str, knowns: dict[str, KnownValue]) -> None:
    if "k" not in knowns and ("coulomb" in text.lower() or "electric" in text.lower()):
        add_known(knowns, "k", "8.9875517923e9", "N")


def add_known(knowns: dict[str, KnownValue], symbol: str, raw_value: str, unit: str) -> None:
    try:
        value = parse_number(raw_value)
    except ValueError:
        return
    si_value, si_unit = convert_value(value, unit)
    knowns[symbol] = KnownValue(symbol, value, unit, si_value, si_unit, f"{raw_value} {unit}".strip())
    if symbol == "U":
        knowns.setdefault("V", KnownValue("V", value, unit, si_value, si_unit, f"{raw_value} {unit}".strip()))


def infer_target(question: str, knowns: dict[str, KnownValue]) -> str:
    text = question.lower()
    if "inductive reactance" in text:
        return "X_L"
    if "capacitive reactance" in text:
        return "X_C"
    if "impedance" in text:
        return "Z"
    if "power" in text or "dissipated" in text:
        return "P"
    if "current" in text:
        return "I"
    if "voltage" in text or "potential" in text:
        return "V"
    if "resistance" in text:
        return "R"
    if "electric field" in text:
        return "E"
    if "force" in text:
        return "F"
    if "energy" in text:
        return "E"
    for symbol in ("P", "Z", "I", "V", "R", "X_L", "X_C"):
        if symbol not in knowns:
            return symbol
    return ""


def infer_target_unit(target: str) -> str:
    return {
        "X_L": "Ω",
        "XL": "Ω",
        "X_C": "Ω",
        "XC": "Ω",
        "Z": "Ω",
        "R": "Ω",
        "Req": "Ω",
        "P": "W",
        "I": "A",
        "V": "V",
        "U": "V",
        "E": "J",
        "F": "N",
        "C": "F",
    }.get(target, "")


def infer_topic(question: str, target: str) -> str:
    text = question.lower()
    if "test charge" in text and "force" in text:
        return "net_coulomb_force"
    if "rlc" in text:
        return "series_rlc_impedance"
    if "inductive reactance" in text:
        return "inductive_reactance"
    if "capacitive reactance" in text:
        return "capacitive_reactance"
    if "impedance" in text:
        return "impedance"
    if "power" in text or "dissipated" in text:
        return "power"
    if "coulomb" in text:
        return "coulomb_force"
    return target.lower()


def normalize_input_symbol(symbol: str) -> str:
    cleaned = clean_symbol(symbol)
    aliases = {"U": "V", "XL": "X_L", "XC": "X_C"}
    return aliases.get(cleaned, cleaned)


def clean_symbol(symbol: str) -> str:
    cleaned = symbol.strip()
    if cleaned.lower() == "f":
        return "f"
    if cleaned.lower().startswith("q") and len(cleaned) > 1:
        return "q" + cleaned[1:]
    return cleaned.upper().replace("_", "_")


def infer_unit_from_symbol(symbol: str) -> str:
    return {
        "R": "Ω",
        "Z": "Ω",
        "V": "V",
        "U": "V",
        "I": "A",
        "P": "W",
        "L": "H",
        "C": "F",
        "f": "Hz",
        "Q": "C",
        "q": "C",
        "q1": "C",
        "q2": "C",
        "r": "m",
        "F": "N",
        "AB": "m",
        "AC": "m",
        "BC": "m",
    }.get(symbol, "")


def normalize_text(question: str) -> str:
    return question.replace("μ", "μ").replace("µ", "µ").replace("Ω", "Ω")


def parse_number(raw_value: str) -> float:
    text = re.sub(r"\s+", "", raw_value)
    match = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)(?:×|x|\*)10\^?([-+]?\d+)", text)
    if match:
        return float(match.group(1)) * 10 ** int(match.group(2))
    return float(text)


def extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    first = stripped.find("{")
    last = stripped.rfind("}")
    candidates = [stripped]
    if first != -1 and last != -1 and first < last:
        candidates.insert(0, stripped[first : last + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None
