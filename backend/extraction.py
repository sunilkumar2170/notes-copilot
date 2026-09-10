"""
extraction.py — Notes Copilot core extraction pipeline (Gemini API)

Takes a raw therapy session transcript, optionally grounded with
retrieved context (documentation guidelines, safety guidelines, prior
session history — see rag.py), and returns a validated SessionExtraction
object: summary, key topics, mood indicators, follow-up actions, and a
risk_flag boolean for clinician review.

Design principles (matching clinical safety & precision requirements):
- Schema-constrained output (no free-form JSON guessing)
- Pydantic validation on the parsed output
- Explicit "do not infer / no diagnosis" instructions to prevent hallucination
- Risk flag is a SIGNAL for a human clinician to review, never an automated clinical judgment
- Grounding context is clearly separated from the primary transcript
- Environment variables loaded cleanly via python-dotenv
"""

import os
import sys
import json
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pydantic import ValidationError
from schemas import SessionExtraction

load_dotenv()

API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# Determine whether to use official modern google-genai SDK or google-generativeai
try:
    from google import genai
    from google.genai import types
    USE_MODERN_SDK = True
    _genai_client = genai.Client(api_key=API_KEY) if API_KEY else None
except ImportError:
    import google.generativeai as legacy_genai
    USE_MODERN_SDK = False
    if API_KEY:
        legacy_genai.configure(api_key=API_KEY)

EXTRACTION_SYSTEM_PROMPT = """You are an expert AI clinical documentation assistant for licensed mental health clinicians.

Your job is to transform a therapy session transcript into an accurate, concise, neutral, and clinician-reviewed structured draft note.

CRITICAL CLINICAL BOUNDARIES:
- You are an assistive documentation tool, NOT a therapist or clinician.
- Do NOT diagnose any medical or psychiatric condition.
- Do NOT recommend or prescribe treatments, medications, or therapies.
- Do NOT make autonomous clinical decisions.
- Do NOT invent or infer facts, symptoms, emotions, life events, or history that are not explicitly stated in the transcript.
- All draft notes require clinician review, edit, and sign-off.

DATA FIDELITY & SOURCE RULES:
1. SESSION TRANSCRIPT IS THE PRIMARY SOURCE OF TRUTH:
   - Everything documented must be directly stated or clearly supported by the current transcript.
2. PREVIOUS SESSION CONTEXT ISOLATION:
   - If grounding reference or prior session context is provided, use it ONLY to recognize established themes.
   - NEVER present events or discussions from prior sessions as having occurred in the current session.
3. EXISTING HABITS VS. NEW FOLLOW-UP ACTIONS:
   - What the client is already doing (e.g. established morning routine, existing coping tools) belongs in the summary as reported progress or current behavior.
   - Follow-up actions MUST ONLY contain concrete, newly agreed or explicitly modified action commitments for the upcoming week.

FIELD SPECIFICATIONS:

1. SUMMARY (2-4 concise, neutral sentences in professional clinical documentation style):
   - Sentence 1: Client's reported presentation, primary focus, or current stressors (e.g. "Client reported feeling overwhelmed by an approaching work deadline and avoiding tasks due to concerns about performance.").
   - Sentence 2: Reported progress, status of existing strategies, or behavioral observations (e.g. "Client reported consistent adherence to morning breathing exercises, while evening journaling remained inconsistent.").
   - Sentence 3: Agreed focus or manageable next steps identified in the session (e.g. "The client agreed to break the project into small steps starting with a 15-minute outline and to message their manager regarding workload.").
   - Maintain objective clinical voice ("Client reported", "Client stated", "Client identified"). Avoid interpretations, moralizing, or dramatic phrasing.

2. KEY TOPICS (2-5 short, distinct labels):
   - Return concise, specific clinical topics actually discussed (e.g., ["task avoidance", "performance anxiety", "workload management"]).
   - Avoid generic or duplicate terms.

3. MOOD INDICATORS (Factual wording from transcript):
   - Objective mood states or emotional expressions explicitly stated by the client or observed (e.g., ["overwhelmed", "avoidant", "mild anxiety", "low mood", "exhaustion"]).
   - If safety/hopelessness was explored and denied, note factually: ["passive hopelessness — no intent or plan reported"].
   - If no distinct mood descriptors are present, return [].

4. FOLLOW-UP ACTIONS (Concrete, agreed commitments only):
   - Include ONLY concrete, attributable tasks, behavioral experiments, or next steps explicitly agreed upon during the session.
   - Format as concise, actionable statements with timeframe if discussed (e.g., "Write project outline within 15 minutes by tomorrow evening", "Send message to manager to schedule workload conversation", "Move journaling to after dinner").
   - Do NOT create follow-up actions from general conversational topics or unchanged existing habits.
   - If no follow-up actions were agreed upon, return [].

5. RISK FLAG (Strict safety triage signal):
   - Set risk_flag to true ONLY if the transcript contains language indicating potential safety concerns (e.g. explicit or passive hopelessness like "what's the point of trying", thoughts of self-harm, suicidal ideation, threats of harm).
   - Set risk_flag to false if no safety or self-harm language is present.
   - A risk flag is strictly an assistive review signal for a clinician, never an automated diagnosis or escalation.

MINIMAL / GREETING / NON-CLINICAL INPUT:
If the transcript consists only of greetings (e.g. "hi", "hello", "hey there"), chit-chat, or contains no substantive clinical content, return:
{
  "summary": "No substantive session content was provided.",
  "key_topics": [],
  "mood_indicators": [],
  "follow_up_actions": [],
  "risk_flag": false
}

OUTPUT FORMAT:
Return ONLY a valid JSON object matching this schema with no markdown formatting or extra text:
{
  "summary": "string",
  "key_topics": ["string"],
  "mood_indicators": ["string"],
  "follow_up_actions": ["string"],
  "risk_flag": false
}
"""


def extract_session_notes(transcript: str, context: str = "") -> SessionExtraction:
    """
    Send a transcript (optionally with grounding context from rag.py's
    build_context()) to Gemini and return a validated SessionExtraction.

    Raises ValueError if the model output is not valid JSON or fails
    Pydantic schema validation.
    """
    cleaned_transcript = transcript.strip()
    if not cleaned_transcript:
        raise ValueError("Transcript text cannot be empty.")

    # Check for trivial greeting / non-clinical input directly to save latency and guarantee consistency
    if cleaned_transcript.lower() in ["hi", "hello", "hey", "test", "hi there", "hello there", "good morning", "good evening"]:
        return SessionExtraction(
            summary="No substantive session content was provided.",
            key_topics=[],
            mood_indicators=[],
            follow_up_actions=[],
            risk_flag=False,
        )

    if context:
        user_content = f"[Grounding Reference Context]\n{context}\n\n---\n\n[Current Session Transcript]\n{cleaned_transcript}"
    else:
        user_content = f"[Current Session Transcript]\n{cleaned_transcript}"

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY or GOOGLE_API_KEY environment variable is not set. "
            "Please set it in your environment or a .env file."
        )

    raw_text = ""

    if USE_MODERN_SDK:
        client = genai.Client(api_key=api_key)
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=user_content,
                config=types.GenerateContentConfig(
                    system_instruction=EXTRACTION_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )
            raw_text = response.text.strip()
        except Exception as e:
            # If specified model is unavailable, attempt standard fallback models
            if "not found" in str(e).lower() or "404" in str(e):
                for fallback_model in ["gemini-2.0-flash", "gemini-1.5-flash"]:
                    try:
                        response = client.models.generate_content(
                            model=fallback_model,
                            contents=user_content,
                            config=types.GenerateContentConfig(
                                system_instruction=EXTRACTION_SYSTEM_PROMPT,
                                response_mime_type="application/json",
                                temperature=0.1,
                            ),
                        )
                        raw_text = response.text.strip()
                        break
                    except Exception:
                        continue
            if not raw_text:
                raise ValueError(f"Gemini API generation failed: {e}")
    else:
        legacy_genai.configure(api_key=api_key)
        model = legacy_genai.GenerativeModel(
            model_name=MODEL_NAME,
            system_instruction=EXTRACTION_SYSTEM_PROMPT,
            generation_config={"response_mime_type": "application/json", "temperature": 0.1},
        )
        try:
            response = model.generate_content(user_content)
            raw_text = response.text.strip()
        except Exception as e:
            raise ValueError(f"Gemini API generation failed: {e}")

    # Strip markdown fence if present
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json\n"):
            raw_text = raw_text[5:]
        elif raw_text.startswith("json"):
            raw_text = raw_text[4:]

    raw_text = raw_text.strip()

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