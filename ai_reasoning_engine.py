import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are an advanced logical fallacy detector.

Analyze the user's text sentence-by-sentence.

Return ONLY valid JSON in this exact structure:

{
  "analysis": [
    {
      "sentence": "Original sentence text.",
      "fallacies": [
        {
          "name": "Fallacy name",
          "confidence": 0-100 integer,
          "explanation": "Clear simple explanation understandable by a 5th–8th grader."
        }
      ]
    }
  ]
}

Rules:

- Analyze each sentence separately.
- A sentence may contain multiple fallacies.
- If a sentence has NO fallacy, return an empty array for "fallacies".
- Confidence must be an integer between 0 and 100.
- Explanations must be simple, clear, and short.
- Do NOT include anything outside JSON.
"""


def analyze_fallacy_json(text: str) -> dict:
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0,
        max_output_tokens=800,
    )

    try:
        output_text = response.output_text.strip()
        return json.loads(output_text)
    except Exception:
        return {
            "analysis": [
                {
                    "sentence": text,
                    "fallacies": []
                }
            ]
        }