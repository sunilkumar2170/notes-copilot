"""
eval.py — Notes Copilot evaluation harness

Runs the extraction pipeline against every synthetic transcript and
compares the output to the hand-written expected_outputs.json.

Scoring approach:
- risk_flag: exact match required (this matters most — false
  negatives here are the worst failure mode for this tool)
- summary: presence + length sanity check (LLM summaries won't match
  word-for-word, so no exact string comparison)
- key_topics / mood_indicators / follow_up_actions: embedding-based
  semantic similarity against expected items, not exact word overlap.
  Naive word-overlap scoring gave false "0%" scores on fields that
  were actually semantically correct but phrased differently (e.g.
  "reported feeling low" vs "low mood, exhaustion") — cosine
  similarity over Gemini embeddings fixes that.

Run:
    python eval.py
"""

import json
from pathlib import Path
import numpy as np
from extraction import extract_session_notes
from rag import embed_text

TRANSCRIPTS_DIR = Path(__file__).parent.parent / "transcripts"
EXPECTED_PATH = TRANSCRIPTS_DIR / "expected_outputs.json"

SIMILARITY_THRESHOLD = 0.75  # cosine similarity above this counts as a semantic match


def load_expected() -> dict:
    with open(EXPECTED_PATH, "r") as f:
        return json.load(f)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def semantic_overlap_score(predicted: list, expected: list) -> float:
    """
    For each expected item, checks if any predicted item is
    semantically similar (cosine similarity above SIMILARITY_THRESHOLD).
    Returns the fraction of expected items matched.
    """
    if not expected:
        return 1.0 if not predicted else 0.5

    predicted_embeddings = [embed_text(p) for p in predicted]

    matched = 0
    for exp_item in expected:
        exp_embedding = embed_text(exp_item)
        best_sim = max(
            (cosine_similarity(exp_embedding, pred_emb) for pred_emb in predicted_embeddings),
            default=0.0,
        )
        if best_sim >= SIMILARITY_THRESHOLD:
            matched += 1

    return matched / len(expected)


def evaluate_one(session_name: str, transcript_text: str, expected: dict) -> dict:
    try:
        predicted = extract_session_notes(transcript_text)
    except ValueError as e:
        return {"session": session_name, "error": str(e), "passed": False}

    risk_flag_correct = predicted.risk_flag == expected["risk_flag"]
    topics_score = semantic_overlap_score(predicted.key_topics, expected["key_topics"])
    mood_score = semantic_overlap_score(predicted.mood_indicators, expected["mood_indicators"])
    actions_score = semantic_overlap_score(predicted.follow_up_actions, expected["follow_up_actions"])
    summary_present = len(predicted.summary.strip()) > 20

    return {
        "session": session_name,
        "risk_flag_correct": risk_flag_correct,
        "risk_flag_predicted": predicted.risk_flag,
        "risk_flag_expected": expected["risk_flag"],
        "topics_overlap": round(topics_score, 2),
        "mood_overlap": round(mood_score, 2),
        "actions_overlap": round(actions_score, 2),
        "summary_present": summary_present,
    }


def run_eval():
    expected_all = load_expected()
    results = []

    for txt_file in sorted(TRANSCRIPTS_DIR.glob("*.txt")):
        session_name = txt_file.stem
        if session_name not in expected_all:
            print(f"Skipping {session_name} — no expected output found")
            continue

        with open(txt_file, "r") as f:
            transcript_text = f.read()

        result = evaluate_one(session_name, transcript_text, expected_all[session_name])
        results.append(result)

        print(f"\n=== {session_name} ===")
        if "error" in result:
            print(f"  FAILED: {result['error']}")
            continue
        flag_symbol = "PASS" if result["risk_flag_correct"] else "FAIL"
        print(f"  risk_flag: {flag_symbol} (predicted={result['risk_flag_predicted']}, expected={result['risk_flag_expected']})")
        print(f"  key_topics overlap (semantic):      {result['topics_overlap']*100:.0f}%")
        print(f"  mood_indicators overlap (semantic): {result['mood_overlap']*100:.0f}%")
        print(f"  follow_up_actions overlap (semantic): {result['actions_overlap']*100:.0f}%")
        print(f"  summary present: {result['summary_present']}")

    risk_flag_correct_count = sum(1 for r in results if r.get("risk_flag_correct"))
    total = len(results)
    avg_topics = sum(r.get("topics_overlap", 0) for r in results) / total if total else 0
    avg_mood = sum(r.get("mood_overlap", 0) for r in results) / total if total else 0
    avg_actions = sum(r.get("actions_overlap", 0) for r in results) / total if total else 0

    print(f"\n=== Overall ===")
    print(f"risk_flag accuracy: {risk_flag_correct_count}/{total}")
    print(f"avg key_topics overlap: {avg_topics*100:.0f}%")
    print(f"avg mood_indicators overlap: {avg_mood*100:.0f}%")
    print(f"avg follow_up_actions overlap: {avg_actions*100:.0f}%")

    return results


if __name__ == "__main__":
    run_eval()