"""Loose JSON extraction helpers for local model responses."""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json_object(text: str) -> dict[str, Any] | None:
    for candidate in candidate_json_objects(text):
        parsed = parse_json_object(candidate)
        if isinstance(parsed, dict):
            return parsed
    return None


def extract_json_like_fields(text: str, keys: tuple[str, ...]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for key in keys:
        value = extract_json_like_string_field(text, key)
        if value is not None:
            parsed[key] = value
    return parsed


def extract_json_like_string_field(text: str, key: str) -> str | None:
    quoted = re.search(
        rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"',
        text,
        re.DOTALL,
    )
    if quoted:
        return decode_json_like_string(quoted.group(1))

    unquoted = re.search(
        rf'"{re.escape(key)}"\s*:\s*([^,\n}}]+)',
        text,
        re.DOTALL,
    )
    if unquoted:
        return unquoted.group(1).strip().strip('"')

    plain = re.search(
        rf"(?im)^\s*{re.escape(key)}\s*[:=]\s*(.+?)\s*$",
        text,
    )
    if plain:
        return plain.group(1).strip().strip('"')

    if key == "explanation":
        block = re.search(
            rf"(?is)^\s*{re.escape(key)}\s*[:=]\s*(.*)\Z",
            text,
        )
        if block:
            return block.group(1).strip()
    return None


def decode_json_like_string(text: str) -> str:
    chars: list[str] = []
    i = 0
    while i < len(text):
        char = text[i]
        if char != "\\" or i + 1 >= len(text):
            chars.append(char)
            i += 1
            continue
        next_char = text[i + 1]
        if next_char in {'"', "\\", "/", "n", "r", "t", "b", "f"}:
            decoded = {
                "n": "\n",
                "r": "\r",
                "t": "\t",
                "b": "\b",
                "f": "\f",
            }.get(next_char, next_char)
            chars.append(decoded)
        else:
            chars.append(char)
            chars.append(next_char)
        i += 2
    return "".join(chars)


def candidate_json_objects(text: str) -> list[str]:
    stripped = text.strip()
    first = stripped.find("{")
    last = stripped.rfind("}")
    candidates = [stripped]
    if first != -1 and last != -1 and first < last:
        candidates.insert(0, stripped[first : last + 1])
    return deduplicate(candidates)


def parse_json_object(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        repaired = repair_json_backslashes(text)
        if repaired != text:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                return None
    return None


def repair_json_backslashes(text: str) -> str:
    chars: list[str] = []
    in_string = False
    escaped = False
    i = 0
    while i < len(text):
        char = text[i]
        if in_string:
            if escaped:
                chars.append(char)
                escaped = False
            elif char == "\\":
                next_char = text[i + 1] if i + 1 < len(text) else ""
                if next_char in {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}:
                    chars.append(char)
                    escaped = True
                else:
                    chars.append("\\")
                    chars.append(char)
            elif char == '"':
                chars.append(char)
                in_string = False
            else:
                chars.append(char)
        else:
            chars.append(char)
            if char == '"':
                in_string = True
        i += 1
    return "".join(chars)


def deduplicate(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
