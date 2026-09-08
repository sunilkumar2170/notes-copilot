"""
extraction.py — Notes Copilot core extraction pipeline (Gemini API)

Takes a raw therapy session transcript, optionally grounded with
retrieved context (documentation guidelines, safety guidelines, prior
session history — see rag.py), and returns a validated SessionExtraction
object: summary, key topics, mood indicators, follow-up actions, and a
risk_flag boolean for clinician review.

Design principles (matching the "systems people can trust" requirement):
- Schema-constrained output (no free-form JSON guessing)
- Pydantic validation on the parsed output — wrong types or missing
  fields fail loudly with a clear error, not a silent bad object
- Explicit "do not infer" instruction to reduce hallucination
- risk_flag is a SIGNAL for a human clinician to review — this tool
  never makes a clinical judgment call or gives advice. It only
  surfaces language patterns for a professional to look at.
- Grounding context (when provided) is clearly labeled and separated
  from the transcript, so the model can't confuse "a rule I should
  follow" with "something the client said."
- No secrets/API keys hardcoded — loaded from environment.
"""

import os
import json
import google.generativeai as genai
from pydantic import ValidationError
from schemas import SessionExtraction

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

EXTRACTION_SYSTEM_PROMPT = """You are a clinical documentation assistant. \
You extract structured information from therapy session transcripts to \
help a licensed clinician write their notes faster. You do NOT provide \
therapeutic advice, diagnoses, or clinical judgments — you only \
summarize and structure what was explicitly said in the transcript.

You may be given grounding context (documentation guidelines, safety \
guidelines, and/or relevant prior session history). Use guidelines to \
shape HOW you write the note. Use prior session history only to note \
genuine recurring themes — never invent continuity that isn't supported \
by the provided context.

Return ONLY valid JSON with this exact schema, and nothing else:
{
  "summary": "2-4 sentence neutral summary of what was discussed",
  "key_topics": ["short phrase", "short phrase"],
  "mood_indicators": ["short phrase describing stated or clearly implied mood"],
  "follow_up_actions": ["concrete action agreed upon in the session"],
  "risk_flag": true or false
}

Rules:
- Do not infer information that was not stated or clearly implied in the transcript.
- risk_flag should be true ONLY if the transcript contains language suggesting \
distress, hopelessness, or safety concerns that a clinician should review \
promptly. This is a flag for human review, not a diagnosis.
- Do not include markdown code fences, explanation, or any text outside the JSON object.
- If a category has no matches, return an empty array.
"""

model = genai.GenerativeModel(
    model_name="gemini-3.6-flash",
    system_instruction=EXTRACTION_SYSTEM_PROMPT,
    generation_config={"response_mime_type": "application/json"},
)


def extract_session_notes(transcript: str, context: str = "") -> SessionExtraction:
    """
    Send a transcript (optionally with grounding context from rag.py's
    build_context()) to Gemini and return a validated SessionExtraction.

    Raises ValueError if the model output is not valid JSON or fails
    Pydantic schema validation (wrong types, missing fields, or a
    suspiciously short/empty summary).
    """
    if context:
        user_content = f"Grounding context:\n{context}\n\n---\n\nTranscript:\n\n{transcript}"
    else:
        user_content = f"Transcript:\n\n{transcript}"

    response = model.generate_content(user_content)

    raw_text = response.text.strip()

    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json\n"):
            raw_text = raw_text[5:]

    try:
        raw_json = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON: {e}\nRaw output: {raw_text}")

    try:
        return SessionExtraction(**raw_json)
    except ValidationError as e:
        raise ValueError(f"Model output failed schema validation:\n{e}\nRaw output: {raw_text}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python extraction.py <path_to_transcript.txt>")
        sys.exit(1)

    with open(sys.argv[1], "r") as f:
        transcript_text = f.read()

    result = extract_session_notes(transcript_text)
    print(result.model_dump_json(indent=2))