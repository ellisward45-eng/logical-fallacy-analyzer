# file: ai_reasoning_engine.py
from __future__ import annotations

import json
import os
from typing import Any, Dict

from dotenv import load_dotenv
from openai import OpenAI


SYSTEM_PROMPT = """
You are a logical fallacy detector.

Return ONLY valid JSON in this format:

{
  "fallacy": "name of fallacy OR none",
  "confidence": "High | Medium | Low",
  "explanation": "Simple explanation a 5th–8th grader can understand."
}

Rules:
- Output MUST be valid JSON only. No extra text.
- If no fallacy exists use exactly:
  {"fallacy":"none","confidence":"Low","explanation":"No logical fallacy detected."}
""".strip()


def _fallback_result() -> Dict[str, str]:
    return {
        "fallacy": "none",
        "confidence": "Low",
        "explanation": "No logical fallacy detected.",
    }


def _normalize_result(obj: Any) -> Dict[str, str]:
    if not isinstance(obj, dict):
        return _fallback_result()

    fallacy = str(obj.get("fallacy", "none")).strip() or "none"
    confidence = str(obj.get("confidence", "Low")).strip() or "Low"
    explanation = str(obj.get("explanation", "")).strip() or "No logical fallacy detected."

    conf_norm = confidence.lower()
    if conf_norm not in {"high", "medium", "low"}:
        confidence = "Low"
    else:
        confidence = conf_norm.capitalize()

    if fallacy == "":
        fallacy = "none"

    return {"fallacy": fallacy, "confidence": confidence, "explanation": explanation}


def _get_client() -> OpenAI:
    load_dotenv()

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Put it in your .env file.")

    project_id = (os.getenv("OPENAI_PROJECT_ID") or "").strip()

    kwargs: Dict[str, Any] = {"api_key": api_key}
    if project_id:
        kwargs["project"] = project_id

    return OpenAI(**kwargs)


_client = _get_client()


def analyze_fallacy_json(text: str) -> Dict[str, str]:
    """
    Analyze a user's statement and return a JSON dict:
    { "fallacy": str, "confidence": "High|Medium|Low", "explanation": str }

    This function ALWAYS returns a dict in that format.
    """
    user_text = (text or "").strip()
    if not user_text:
        return _fallback_result()

    try:
        resp = _client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            temperature=0,
            max_output_tokens=256,
            response_format={"type": "json_object"},
        )

        raw = (resp.output_text or "").strip()
        if not raw:
            return _fallback_result()

        data = json.loads(raw)
        return _normalize_result(data)

    except Exception:
        return _fallback_result()