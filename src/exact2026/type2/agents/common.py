"""Common helpers for LangChain-backed Type 2 agents."""

from __future__ import annotations

import json
from typing import Any

from ..llm import message_content
from ..json_utils import extract_json_object


def invoke_json_agent(llm: Any, system_prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
    if llm is None:
        return {}
    user_text = json.dumps(payload, ensure_ascii=False)
    try:
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", "{payload}")]
        )
        response = (prompt | llm).invoke({"payload": user_text})
    except Exception:
        try:
            response = llm.invoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ]
            )
        except Exception:
            return {}
    return extract_json_object(message_content(response)) or {}
