"""
tools.py — Notes Copilot agent + tool-calling layer

Takes the structured extraction output and decides which actions to
take. This is intentionally a simple rule-based dispatcher rather than
a full LLM agent loop — for this use case (clear, bounded actions
based on structured fields we already extracted), an LLM re-deciding
"what tool to call" would just add latency and hallucination risk for
no benefit. The extraction step already did the reasoning; this layer
just acts on it.

Each tool is a plain function so it's also trivially exposable via MCP
(see mcp_server.py) without any wrapper logic duplicated.
"""

<<<<<<< HEAD
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

=======
from datetime import datetime, timedelta
>>>>>>> 039a93fc7a3d11585e0d76b1ffc6027cc3b3e691
from schemas import SessionExtraction, NoteToolResult, RiskToolResult, FollowupToolResult, AgentOutput


def schedule_followup_tool(actions: list, session_date: str = None) -> FollowupToolResult:
    """
    Given a list of follow-up actions from extraction, produce a
    structured follow-up record. In a real system this would call a
    calendar/EMR API — here it returns the structured payload that
    integration would consume.
    """
    if session_date is None:
        session_date = datetime.now().strftime("%Y-%m-%d")

    next_check_in = (
        datetime.strptime(session_date, "%Y-%m-%d") + timedelta(days=7)
    ).strftime("%Y-%m-%d")

    return FollowupToolResult(
        tool="schedule_followup",
        session_date=session_date,
        suggested_next_check_in=next_check_in,
        actions=actions,
        status="queued_for_integration",
    )


def flag_risk_tool(risk_flag: bool, mood_indicators: list, session_name: str = "") -> RiskToolResult:
    """
    Surfaces a risk flag for clinician review. This tool NEVER makes a
    clinical decision — it only creates a review item. A human clinician
    is always the one who acts on it.
    """
    if not risk_flag:
        return RiskToolResult(tool="flag_risk", flagged=False)

    return RiskToolResult(
        tool="flag_risk",
        flagged=True,
        reason="Language suggesting distress or safety concern detected — flagged for clinician review, not auto-escalated.",
        mood_indicators=mood_indicators,
        status="pending_clinician_review",
    )


def generate_note_tool(summary: str, key_topics: list) -> NoteToolResult:
    """
    Formats the extracted summary into a clean clinical note draft.
    A clinician reviews and edits this before it becomes the official
    record — this tool produces a draft, not a final note.
    """
    note = f"Session Summary (draft — pending clinician review):\n\n{summary}\n\nTopics discussed: {', '.join(key_topics)}"
    return NoteToolResult(
        tool="generate_note",
        draft_note=note,
        status="draft_pending_review",
    )


def run_agent(extraction_result: SessionExtraction, session_name: str = "") -> AgentOutput:
    """
    Orchestrator: given a validated SessionExtraction, decides which
    tools to invoke and returns their combined output as an AgentOutput.
    """
    note = generate_note_tool(extraction_result.summary, extraction_result.key_topics)
    risk = flag_risk_tool(extraction_result.risk_flag, extraction_result.mood_indicators, session_name)

    followup = None
    if extraction_result.follow_up_actions:
        followup = schedule_followup_tool(extraction_result.follow_up_actions)

    return AgentOutput(note=note, risk=risk, followup=followup)


if __name__ == "__main__":
    import sys
    from extraction import extract_session_notes

    if len(sys.argv) != 2:
        print("Usage: python tools.py <path_to_transcript.txt>")
        sys.exit(1)

    with open(sys.argv[1], "r") as f:
        transcript_text = f.read()

    extraction = extract_session_notes(transcript_text)
    agent_output = run_agent(extraction, session_name=sys.argv[1])
    print(agent_output.model_dump_json(indent=2))