"""Prompt loading helpers for Type 2 agents."""

from __future__ import annotations

from importlib.resources import files


def load_prompt(name: str) -> str:
    return files(__package__).joinpath(name).read_text(encoding="utf-8").strip()
