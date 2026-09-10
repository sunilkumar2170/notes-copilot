"""
main.py — Notes Copilot FastAPI Clinical Dashboard

Full pipeline: RAG grounding -> extraction -> agent tools.
Provides a modern clinical dashboard for therapists and clinicians to review
and finalize AI-generated session notes.

Run locally:
    uvicorn main:app --reload

Deploy:
    uvicorn main:app --host 0.0.0.0 --port $PORT
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from extraction import extract_session_notes
from rag import index_documents, retrieve_relevant_context, build_context
from tools import run_agent
from schemas import PipelineResult

load_dotenv()

app = FastAPI(title="Notes Copilot - Clinical Documentation Assistant")

BASE_DIR = Path(__file__).parent.parent
KNOWLEDGE_DIR = str(BASE_DIR / "knowledge")
TRANSCRIPTS_DIR = BASE_DIR / "transcripts"

_rag_index = None
_rag_chunks = None


def get_rag_index():
    global _rag_index, _rag_chunks
    if _rag_index is None:
        _rag_index, _rag_chunks = index_documents(KNOWLEDGE_DIR)
    return _rag_index, _rag_chunks


def process_transcript(transcript: str) -> PipelineResult:
    """Full pipeline: retrieve grounding context -> extract -> run agent tools."""
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


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Clinical Notes Copilot | AI Documentation Assistant</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-page: #f8fafc;
            --surface: #ffffff;
            --border: #e2e8f0;
            --border-hover: #cbd5e1;
            --primary: #2563eb;
            --primary-hover: #1d4ed8;
            --primary-light: #eff6ff;
            --primary-text: #1e40af;
            --text-main: #0f172a;
            --text-secondary: #475569;
            --text-muted: #64748b;
            --danger: #dc2626;
            --danger-bg: #fef2f2;
            --danger-border: #fecaca;
            --danger-text: #991b1b;
            --safe: #059669;
            --safe-bg: #ecfdf5;
            --safe-border: #a7f3d0;
            --safe-text: #065f46;
            --warning: #d97706;
            --warning-bg: #fffbeb;
            --warning-border: #fde68a;
            --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
            --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.07), 0 2px 4px -2px rgb(0 0 0 / 0.07);
            --radius-md: 10px;
            --radius-lg: 14px;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-page);
            color: var(--text-main);
            line-height: 1.5;
            padding: 32px 16px 80px;
        }

        .container {
            max-width: 880px;
            margin: 0 auto;
        }

        /* Header */
        .header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 24px;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border);
        }

        .brand-title {
            font-size: 22px;
            font-weight: 700;
            color: var(--text-main);
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .brand-icon {
            background: var(--primary);
            color: white;
            width: 32px;
            height: 32px;
            border-radius: 8px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
        }

        .brand-subtitle {
            font-size: 13px;
            color: var(--text-muted);
            margin-top: 2px;
        }

        .system-badge {
            background: #f1f5f9;
            color: #334155;
            font-size: 12px;
            font-weight: 600;
            padding: 4px 12px;
            border-radius: 20px;
            border: 1px solid var(--border);
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }

        .pulse-dot {
            width: 8px;
            height: 8px;
            background: #10b981;
            border-radius: 50%;
            display: inline-block;
        }

        /* Clinical Banner */
        .disclaimer-banner {
            background: var(--warning-bg);
            border: 1px solid var(--warning-border);
            border-radius: var(--radius-md);
            padding: 12px 16px;
            font-size: 13px;
            color: #92400e;
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        /* Card container */
        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: var(--shadow-sm);
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }

        .card-title {
            font-size: 15px;
            font-weight: 700;
            color: var(--text-main);
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        /* Presets */
        .preset-bar {
            margin-bottom: 14px;
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 8px;
        }

        .preset-label {
            font-size: 12px;
            font-weight: 600;
            color: var(--text-muted);
            margin-right: 4px;
        }

        .btn-preset {
            background: #f8fafc;
            border: 1px solid var(--border);
            color: #334155;
            font-size: 12px;
            font-weight: 500;
            padding: 6px 12px;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.15s ease;
        }

        .btn-preset:hover {
            background: var(--primary-light);
            border-color: var(--primary);
            color: var(--primary-text);
        }

        /* Textarea */
        textarea {
            width: 100%;
            height: 200px;
            font-family: 'JetBrains Mono', Consolas, monospace;
            font-size: 13px;
            line-height: 1.6;
            padding: 14px;
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            resize: vertical;
            background: #ffffff;
            color: var(--text-main);
            transition: border 0.15s, box-shadow 0.15s;
        }

        textarea:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12);
        }

        /* Buttons */
        .action-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 14px;
        }

        .btn-primary {
            background: var(--primary);
            color: white;
            border: none;
            padding: 10px 24px;
            font-size: 14px;
            font-weight: 600;
            border-radius: var(--radius-md);
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            transition: background 0.15s ease;
        }

        .btn-primary:hover {
            background: var(--primary-hover);
        }

        .btn-ghost {
            background: transparent;
            border: 1px solid transparent;
            color: var(--text-muted);
            font-size: 13px;
            padding: 8px 14px;
            border-radius: var(--radius-md);
            cursor: pointer;
        }

        .btn-ghost:hover {
            color: var(--text-main);
            background: #f1f5f9;
        }

        /* Results Display */
        .status-alert {
            padding: 16px 20px;
            border-radius: var(--radius-md);
            margin-bottom: 20px;
            display: flex;
            align-items: flex-start;
            gap: 14px;
        }

        .status-alert.danger {
            background: var(--danger-bg);
            border: 1px solid var(--danger-border);
            color: var(--danger-text);
        }

        .status-alert.safe {
            background: var(--safe-bg);
            border: 1px solid var(--safe-border);
            color: var(--safe-text);
        }

        .alert-icon {
            font-size: 20px;
            line-height: 1;
        }

        .alert-heading {
            font-weight: 700;
            font-size: 14px;
            margin-bottom: 2px;
        }

        .alert-desc {
            font-size: 13px;
            opacity: 0.92;
        }

        /* Section layout */
        .section-block {
            margin-bottom: 22px;
        }

        .section-label {
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .note-container {
            background: #f8fafc;
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 18px 20px;
            position: relative;
        }

        .note-text {
            font-size: 14px;
            line-height: 1.65;
            color: #1e293b;
            white-space: pre-wrap;
        }

        .btn-copy {
            position: absolute;
            top: 12px;
            right: 12px;
            background: #ffffff;
            border: 1px solid var(--border);
            font-size: 12px;
            font-weight: 600;
            color: var(--text-secondary);
            padding: 4px 10px;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.15s;
        }

        .btn-copy:hover {
            background: var(--primary-light);
            border-color: var(--primary);
            color: var(--primary-text);
        }

        /* Chips */
        .chips-wrap {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .chip {
            display: inline-flex;
            align-items: center;
            padding: 5px 12px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 500;
        }

        .chip-topic {
            background: #eff6ff;
            color: #1d4ed8;
            border: 1px solid #dbeafe;
        }

        .chip-mood {
            background: #f1f5f9;
            color: #334155;
            border: 1px solid #e2e8f0;
        }

        /* Action Checklist */
        .action-list {
            list-style: none;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .action-item {
            display: flex;
            align-items: flex-start;
            gap: 12px;
            background: #f8fafc;
            border: 1px solid var(--border);
            padding: 12px 14px;
            border-radius: 8px;
            font-size: 13.5px;
            color: #1e293b;
        }

        .action-item input[type="checkbox"] {
            margin-top: 3px;
            accent-color: var(--primary);
            width: 16px;
            height: 16px;
            cursor: pointer;
        }

        /* Grounding / Evidence */
        .grounding-box {
            background: #f8fafc;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px 16px;
            font-size: 12.5px;
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
        }

        .grounding-tag {
            background: #e2e8f0;
            color: #1e293b;
            font-family: 'JetBrains Mono', Consolas, monospace;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 11.5px;
            font-weight: 500;
        }

        /* Collapsible Raw JSON */
        details {
            margin-top: 20px;
            border-top: 1px solid var(--border);
            padding-top: 16px;
        }

        summary {
            cursor: pointer;
            font-size: 13px;
            font-weight: 600;
            color: var(--text-muted);
            user-select: none;
        }

        summary:hover {
            color: var(--text-main);
        }

        pre {
            background: #0f172a;
            color: #e2e8f0;
            padding: 16px;
            border-radius: 8px;
            font-family: 'JetBrains Mono', Consolas, monospace;
            font-size: 12px;
            line-height: 1.5;
            overflow-x: auto;
            margin-top: 10px;
        }

        .error-alert {
            background: var(--danger-bg);
            border: 1px solid var(--danger-border);
            color: var(--danger-text);
            padding: 16px;
            border-radius: var(--radius-md);
            font-size: 14px;
        }

        /* Loading indicator */
        .loading-spinner {
            display: none;
            width: 18px;
            height: 18px;
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header class="header">
            <div>
                <div class="brand-title">
                    <span class="brand-icon">⚕</span>
                    Clinical Notes Copilot
                </div>
                <div class="brand-subtitle">AI-Assisted Documentation & Clinical Triage Support</div>
            </div>
            <div class="system-badge">
                <span class="pulse-dot"></span>
                RAG Pipeline Active
            </div>
        </header>

        <!-- Clinical Safety Notice -->
        <div class="disclaimer-banner">
            <span>🛡️</span>
            <div>
                <strong>Clinician Oversight Required:</strong> AI-generated notes and triage signals are assistive drafts and do not replace professional clinical judgment. Licensed clinician review and sign-off is required before finalizing.
            </div>
        </div>

        <!-- Input Form Card -->
        <div class="card">
            <div class="card-header">
                <span class="card-title">Session Transcript Input</span>
            </div>

            <!-- Quick Sample Presets -->
            <div class="preset-bar">
                <span class="preset-label">Load Test Preset:</span>
                <button type="button" class="btn-preset" onclick="loadSample(1)">Session 1: Routine Follow-up</button>
                <button type="button" class="btn-preset" onclick="loadSample(2)">Session 2: Workload & Avoidance</button>
                <button type="button" class="btn-preset" onclick="loadSample(3)">Session 3: Safety Risk Signal</button>
            </div>

            <form id="extractForm" method="post" action="/extract-form" onsubmit="handleSubmit()">
                <textarea id="transcriptInput" name="transcript" placeholder="Paste therapy session transcript here..." required>__TRANSCRIPT__</textarea>
                <div class="action-row">
                    <button type="button" class="btn-ghost" onclick="clearInput()">Clear</button>
                    <button type="submit" class="btn-primary" id="submitBtn">
                        <span class="loading-spinner" id="spinner"></span>
                        <span id="btnText">Generate Clinical Note</span>
                    </button>
                </div>
            </form>
        </div>

        <!-- Extraction Results -->
        __RESULT__
    </div>

    <script>
        const samples = {
            1: `Session Date: 2026-08-14\\nSession Type: Individual Therapy — Follow-up (Session 4)\\n\\nTherapist: How has this past week been for you since we last spoke?\\n\\nClient: It's been okay, honestly. Better than the week before. I've been feeling a bit anxious in the mornings, especially before work, but it eases off once I actually get started on things.\\n\\nTherapist: That's useful to notice — the anxiety being tied to anticipation rather than the work itself. Have you been able to use any of the grounding techniques we talked about?\\n\\nClient: I tried the breathing exercise a couple of times. It helped a little. I forgot about it most days though, I think because I was rushing in the morning.\\n\\nTherapist: That makes sense. Let's think about a cue you could attach it to — something you already do every morning, so it becomes automatic rather than something you have to remember.\\n\\nClient: Maybe right when I make coffee? I do that every single day without fail.\\n\\nTherapist: That sounds like a strong anchor. Let's try that this week — the breathing exercise while your coffee brews, just two minutes.\\n\\nClient: Okay, I can do that.\\n\\nTherapist: How has your sleep been?\\n\\nClient: About the same. Falling asleep is fine, but I still wake up around 4am sometimes and can't get back to sleep right away.\\n\\nTherapist: We'll keep an eye on that. For now, let's set journaling as a follow-up — just a few lines before bed about what's on your mind, so we have something concrete to look at next session alongside the breathing exercise.\\n\\nClient: Sure, I can try that.\\n\\nTherapist: Great. Let's plan to check in on both of these — the morning breathing routine and the evening journaling — same time next week.`,
            2: `Session Date: 2026-08-21\\nSession Type: Individual Therapy — Follow-up (Session 5)\\n\\nTherapist: Let's start with the breathing exercise and journaling from last week — how did those go?\\n\\nClient: The coffee breathing thing actually worked really well, I did it almost every day. The journaling I only did twice, I kept forgetting at night.\\n\\nTherapist: That's great progress on the breathing side. For the journaling, let's move it earlier — maybe right after dinner instead of before bed, when you're less tired.\\n\\nClient: That could work better, yeah.\\n\\nTherapist: Let's also talk about work. You mentioned last time there's a project deadline coming up.\\n\\nClient: Yes, it's next Friday. I'm feeling pretty overwhelmed about it, I haven't started the main part yet.\\n\\nTherapist: What's making it hard to start?\\n\\nClient: I think I'm scared it won't be good enough, so I keep avoiding it entirely.\\n\\nTherapist: That avoidance pattern sounds familiar to what we discussed with the anxiety before work in general. Let's break the project into smaller pieces — what's the very first small step, something that takes fifteen minutes or less?\\n\\nClient: I guess just opening the document and writing an outline.\\n\\nTherapist: Let's set that as a concrete task — outline only, fifteen minutes, by tomorrow evening. We can build from there.\\n\\nClient: Okay, that feels doable.\\n\\nTherapist: One more thing — you mentioned wanting to talk to your manager about workload. Have you had that conversation yet?\\n\\nClient: No, I've been putting it off.\\n\\nTherapist: Let's set that as a follow-up too — even just scheduling the conversation with your manager counts as progress, doesn't have to be the full conversation this week.\\n\\nClient: Alright, I'll aim to at least send the message asking for time to talk.\\n\\nTherapist: Good. So for next week: continue the morning breathing, move journaling to after dinner, write the project outline by tomorrow, and message your manager to schedule a workload conversation.`,
            3: `Session Date: 2026-08-28\\nSession Type: Individual Therapy — Follow-up (Session 6)\\n\\nTherapist: How have things been since we last spoke?\\n\\nClient: Honestly, this week was rough. The work deadline got extended, which should have helped, but I've just felt really low the whole week. Like everything feels pointless right now.\\n\\nTherapist: I hear you — that sounds heavy. When you say everything feels pointless, can you tell me more about what that's been like day to day?\\n\\nClient: I just feel exhausted all the time, even when I'm not doing anything. I've been having thoughts like "what's even the point of trying," especially at night. I'm not doing anything about it, it's just this constant background feeling.\\n\\nTherapist: Thank you for telling me that directly — I want to check in more carefully. When you have those thoughts, do they ever turn into thoughts of wanting to hurt yourself, or wanting to not be here anymore?\\n\\nClient: No, not like that. It's more just... tired of everything, not that I want to act on anything. I don't have any plan or anything like that.\\n\\nTherapist: Okay, I appreciate you being clear about that, and I believe you. I still want us to keep a close eye on this together, and I'd like us to increase our check-ins to twice a week for now instead of once. Would that be okay?\\n\\nClient: Yeah, I think that would help.\\n\\nTherapist: I'd also like you to keep a simple daily note — just a number from 1 to 10 for how you're feeling, so we can track the pattern over the next two weeks. And if at any point those thoughts do shift toward wanting to act on them, I want you to reach out immediately, not wait for our next session.\\n\\nClient: Okay, I understand.\\n\\nTherapist: Let's also loop in the psychiatry team for a check on your sleep and low mood, since this has been going on for a few weeks now. I'll send that referral today.\\n\\nClient: That sounds good, thank you.`
        };

        function loadSample(id) {
            const input = document.getElementById('transcriptInput');
            if (samples[id]) {
                input.value = samples[id];
            }
        }

        function clearInput() {
            document.getElementById('transcriptInput').value = '';
        }

        function handleSubmit() {
            document.getElementById('spinner').style.display = 'inline-block';
            document.getElementById('btnText').innerText = 'Processing with RAG & Gemini...';
            document.getElementById('submitBtn').disabled = true;
        }

        function copyNote() {
            const text = document.getElementById('draftNoteText').innerText;
            navigator.clipboard.writeText(text).then(() => {
                const copyBtn = document.getElementById('copyBtn');
                copyBtn.innerText = '✓ Copied to Clipboard';
                setTimeout(() => {
                    copyBtn.innerText = 'Copy Draft Note';
                }, 2000);
            });
        }
    </script>
</body>
</html>
"""


def render_page(transcript: str = "", result_html: str = "") -> str:
    return PAGE_TEMPLATE.replace("__TRANSCRIPT__", transcript).replace("__RESULT__", result_html)


def render_result(result: PipelineResult) -> str:
    extraction = result.extraction

    if extraction.risk_flag:
        flag_banner = """
        <div class="status-alert danger">
            <span class="alert-icon">⚠️</span>
            <div>
                <div class="alert-heading">Safety Triage Signal: Language of Distress Detected</div>
                <div class="alert-desc">Transcript contains language of hopelessness or potential safety concern. Flagged for immediate clinician review. (Assistive signal only — no automated clinical action taken).</div>
            </div>
        </div>
        """
    else:
        flag_banner = """
        <div class="status-alert safe">
            <span class="alert-icon">✓</span>
            <div>
                <div class="alert-heading">Routine Session Review: No Safety Risk Language Detected</div>
                <div class="alert-desc">Transcript does not contain self-harm or acute distress triggers. Standard clinician review recommended.</div>
            </div>
        </div>
        """

    topics_html = "".join(f'<span class="chip chip-topic">{t}</span>' for t in extraction.key_topics)
    mood_html = "".join(f'<span class="chip chip-mood">{m}</span>' for m in extraction.mood_indicators)

    actions_items = []
    for action in extraction.follow_up_actions:
        actions_items.append(f"""
        <li class="action-item">
            <input type="checkbox" checked id="act_{hash(action)}">
            <label for="act_{hash(action)}">{action}</label>
        </li>
        """)
    actions_html = "".join(actions_items) if actions_items else '<li class="action-item"><span>No concrete follow-up actions recorded for this session.</span></li>'

    grounding_html = "".join(f'<span class="grounding-tag">{g}</span>' for g in result.grounding_used) or "<span>Standard clinical guidelines</span>"

    return f"""
    <div class="card">
        <div class="card-header">
            <span class="card-title">Structured Clinical Draft</span>
            <span style="font-size:12px; color:var(--text-muted); font-weight:600;">Status: Draft Pending Review</span>
        </div>

        {flag_banner}

        <!-- Summary / Draft Note -->
        <div class="section-block">
            <div class="section-label">
                <span>Clinical Summary (Draft)</span>
            </div>
            <div class="note-container">
                <button type="button" class="btn-copy" id="copyBtn" onclick="copyNote()">Copy Draft Note</button>
                <div class="note-text" id="draftNoteText">{extraction.summary}</div>
            </div>
        </div>

        <!-- Key Topics & Mood Indicators -->
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 22px;">
            <div>
                <div class="section-label">Key Topics Identified</div>
                <div class="chips-wrap">{topics_html or '<span style="color:var(--text-muted);font-size:13px;">None identified</span>'}</div>
            </div>
            <div>
                <div class="section-label">Mood & Affect Indicators</div>
                <div class="chips-wrap">{mood_html or '<span style="color:var(--text-muted);font-size:13px;">None identified</span>'}</div>
            </div>
        </div>

        <!-- Follow-up Action Plan -->
        <div class="section-block">
            <div class="section-label">
                <span>Agreed Follow-Up Action Plan & Next Steps</span>
                <span style="font-size:11px; text-transform:none; font-weight:500; color:var(--text-muted);">Checkboxes for clinician review/sign-off</span>
            </div>
            <ul class="action-list">{actions_html}</ul>
        </div>

        <!-- RAG Grounding Evidence -->
        <div class="section-block">
            <div class="section-label">RAG Knowledge Grounding & Evidence</div>
            <div class="grounding-box">
                <span style="font-weight:600;">Referenced Protocols:</span>
                {grounding_html}
            </div>
        </div>

        <!-- Collapsible Raw JSON -->
        <details>
            <summary>View Validated JSON & Agent Tool Payloads</summary>
            <pre>{result.model_dump_json(indent=2)}</pre>
        </details>
    </div>
    """


@app.get("/", response_class=HTMLResponse)
def home():
    return render_page(transcript="", result_html="")


@app.post("/extract-form", response_class=HTMLResponse)
def extract_form(transcript: str = Form(...)):
    try:
        result = process_transcript(transcript)
        result_html = render_result(result)
    except Exception as e:
        result_html = f'<div class="card"><div class="error-alert"><strong>Extraction Failed:</strong> {e}</div></div>'

    return render_page(transcript=transcript, result_html=result_html)


@app.post("/extract")
def extract_api(transcript: str = Form(...)):
    """JSON API endpoint — full pipeline output (extraction + agent actions + grounding used)."""
    return process_transcript(transcript)


@app.get("/health")
def health():
    return {"status": "ok", "sdk": "google-genai", "rag": "faiss"}