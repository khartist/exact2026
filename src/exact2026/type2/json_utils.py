"""Loose JSON extraction helpers for local model responses."""

from __future__ import annotations

import json
import re
from typing import Any


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
