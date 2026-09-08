"""
main.py — Notes Copilot FastAPI app

Single-file backend + frontend for fast, clean deployment.
Wires the full pipeline: RAG grounding -> extraction -> agent tools.

Run locally:
    uvicorn main:app --reload

Deploy (Render):
    Start command -> uvicorn main:app --host 0.0.0.0 --port $PORT
"""

import os
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from extraction import extract_session_notes
from rag import index_documents, retrieve_relevant_context, build_context
from tools import run_agent
from schemas import PipelineResult

app = FastAPI(title="Notes Copilot")

KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "..", "knowledge")

_rag_index = None
_rag_chunks = None


def get_rag_index():
    global _rag_index, _rag_chunks
    if _rag_index is None:
        _rag_index, _rag_chunks = index_documents(KNOWLEDGE_DIR)
    return _rag_index, _rag_chunks


def process_transcript(transcript: str) -> PipelineResult:
    """Full pipeline: retrieve grounding context -> extract -> run agent."""
    index, chunks = get_rag_index()
    relevant = retrieve_relevant_context(index, chunks, transcript, k=3)
    context = build_context(relevant)

    extraction = extract_session_notes(transcript, context=context)
    agent_output = run_agent(extraction)

    return PipelineResult(
        extraction=extraction,
        agent_actions=agent_output,
        grounding_used=[r["source"] for r in relevant],
    )


PAGE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Notes Copilot</title>
    <meta charset="UTF-8">
    <style>
        :root {{
            --primary: #4f46e5;
            --primary-dark: #4338ca;
            --danger: #dc2626;
            --danger-bg: #fef2f2;
            --safe-bg: #f0fdf4;
            --safe-text: #166534;
            --border: #e5e7eb;
            --text-muted: #6b7280;
            --bg: #f9fafb;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            max-width: 780px;
            margin: 0 auto;
            padding: 32px 20px 60px;
            background: var(--bg);
            color: #111827;
        }}
        h2 {{ font-size: 26px; margin-bottom: 4px; }}
        .subtitle {{ color: var(--text-muted); font-size: 14px; margin-bottom: 24px; }}
        .card {{
            background: white;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 20px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        }}
        textarea {{
            width: 100%;
            height: 220px;
            font-family: 'SF Mono', Consolas, monospace;
            font-size: 13px;
            padding: 12px;
            border: 1px solid var(--border);
            border-radius: 8px;
            resize: vertical;
        }}
        textarea:focus {{ outline: 2px solid var(--primary); outline-offset: -1px; }}
        button {{
            margin-top: 12px;
            padding: 10px 22px;
            font-size: 14px;
            font-weight: 600;
            color: white;
            background: var(--primary);
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: background 0.15s;
        }}
        button:hover {{ background: var(--primary-dark); }}
        .section-title {{
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin: 0 0 8px;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 16px;
        }}
        .badge-danger {{ background: var(--danger-bg); color: var(--danger); }}
        .badge-safe {{ background: var(--safe-bg); color: var(--safe-text); }}
        .chip {{
            display: inline-block;
            background: #eef2ff;
            color: var(--primary-dark);
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 13px;
            margin: 2px 4px 2px 0;
        }}
        .grounding-list {{
            font-size: 13px;
            color: var(--text-muted);
        }}
        .grounding-list code {{
            background: #f3f4f6;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 12px;
        }}
        ul.actions {{ margin: 0; padding-left: 20px; }}
        ul.actions li {{ margin-bottom: 6px; }}
        .note-box {{
            background: #fafafa;
            border: 1px dashed var(--border);
            border-radius: 8px;
            padding: 14px 16px;
            font-size: 14px;
            line-height: 1.6;
            white-space: pre-wrap;
        }}
        .pending-tag {{
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            color: #d97706;
            background: #fffbeb;
            padding: 2px 8px;
            border-radius: 4px;
            margin-left: 8px;
        }}
        details {{ margin-top: 16px; }}
        summary {{
            cursor: pointer;
            font-size: 13px;
            color: var(--text-muted);
            font-weight: 600;
        }}
        pre {{
            background: #111827;
            color: #d1d5db;
            padding: 16px;
            border-radius: 8px;
            font-size: 12px;
            overflow-x: auto;
            margin-top: 8px;
        }}
        .error-box {{
            background: var(--danger-bg);
            color: var(--danger);
            padding: 14px 16px;
            border-radius: 8px;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <h2>Notes Copilot</h2>
    <p class="subtitle">Paste a session transcript to generate a structured, clinician-reviewed draft note.</p>

    <div class="card">
        <form method="post" action="/extract-form">
            <textarea name="transcript" placeholder="Paste transcript here...">{transcript}</textarea><br>
            <button type="submit">Extract Notes</button>
        </form>
    </div>

    {result}
</body>
</html>
"""


def render_result(result: PipelineResult) -> str:
    extraction = result.extraction

    if extraction.risk_flag:
        flag_badge = '<span class="badge badge-danger">⚠ Flagged for clinician review</span>'
    else:
        flag_badge = '<span class="badge badge-safe">No risk flag</span>'

    topics_html = "".join(f'<span class="chip">{t}</span>' for t in extraction.key_topics)
    mood_html = "".join(f'<span class="chip">{m}</span>' for m in extraction.mood_indicators)
    actions_html = "".join(f"<li>{a}</li>" for a in extraction.follow_up_actions) or "<li>None recorded</li>"
    grounding_html = ", ".join(f"<code>{g}</code>" for g in result.grounding_used) or "none"

    return f"""
    <div class="card">
        {flag_badge}
        <div class="section-title">Summary</div>
        <div class="note-box">{extraction.summary}</div>

        <div class="section-title" style="margin-top:20px;">Key Topics</div>
        <div>{topics_html or '<span style="color:#9ca3af;font-size:13px;">None identified</span>'}</div>

        <div class="section-title" style="margin-top:20px;">Mood Indicators</div>
        <div>{mood_html or '<span style="color:#9ca3af;font-size:13px;">None identified</span>'}</div>

        <div class="section-title" style="margin-top:20px;">Follow-up Actions <span class="pending-tag">Pending Approval</span></div>
        <ul class="actions">{actions_html}</ul>

        <div class="section-title" style="margin-top:20px;">Grounding Used</div>
        <div class="grounding-list">{grounding_html}</div>

        <details>
            <summary>View raw JSON output</summary>
            <pre>{result.model_dump_json(indent=2)}</pre>
        </details>
    </div>
    """


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE_TEMPLATE.format(transcript="", result="")


@app.post("/extract-form", response_class=HTMLResponse)
def extract_form(transcript: str = Form(...)):
    try:
        result = process_transcript(transcript)
        result_html = render_result(result)
    except ValueError as e:
        result_html = f'<div class="card"><div class="error-box">Extraction failed: {e}</div></div>'

    return PAGE_TEMPLATE.format(transcript=transcript, result=result_html)


@app.post("/extract")
def extract_api(transcript: str = Form(...)):
    """JSON API endpoint — full pipeline output (extraction + agent actions + grounding used)."""
    return process_transcript(transcript)


@app.get("/health")
def health():
    return {"status": "ok"}