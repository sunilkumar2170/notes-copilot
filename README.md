# Notes Copilot

A tool that turns a therapy session transcript into a structured summary
and follow-up action list, grounded in documentation guidelines, safety
guidelines, and relevant prior session context via RAG. Flags anything
a clinician should review and exposes everything as MCP tools.
Built for the Amaha AI role challenge.

## Problem

Clinicians spend time on documentation that pulls them away from care.
This tool takes a raw session transcript and produces a structured,
guideline-consistent draft note plus follow-up actions — reviewed and
finalized by the clinician, not auto-published.

## Architecture

```
transcript.txt
      |
      v
rag.py  --> retrieves grounding context from knowledge/:
              - documentation_guidelines (how to write the note)
              - safety_guidelines (how to handle risk-flagged content)
              - synthetic_previous_sessions (recurring-theme continuity)
      |
      v
extraction.py  --> structured JSON (summary, topics, mood, follow-ups,
                    risk_flag) via Gemini, grounded by the retrieved
                    context, schema-constrained output
      |
      v
tools.py --> agent decides + executes: generate_note, flag_risk, schedule_followup
      |
      v
main.py (FastAPI)  --------------------  mcp_server.py (MCP tools)
   web form + JSON API                    same pipeline, exposed to MCP clients
```

## Why RAG has a real purpose here (not just "FAISS + embeddings")

The knowledge base is split into three categories with distinct jobs:

- **`documentation_guidelines/`** — grounds HOW the note should be
  written (neutral language, attributable follow-ups, recurring-theme
  visibility). Retrieved context shapes note quality, not content.
- **`safety_guidelines/`** — grounds what risk_flag means and, just
  as importantly, what the tool must NOT do (no auto-escalation, no
  clinical judgment, no advice). This is retrieved so the safety
  boundary is reinforced at generation time, not just documented in
  a README no one reads at inference time.
- **`synthetic_previous_sessions/`** — grounds recurring-theme
  detection (e.g. "sleep has come up before") without inventing
  continuity that isn't actually supported by real prior context.

`rag.py` exposes three functions with one job each:
`index_documents()`, `retrieve_relevant_context()` (optionally
filtered by category), and `build_context()` (formats retrieved
chunks into a labeled, prompt-ready string so the model never
confuses "a rule" with "something the client said").

## Why the rest of the design looks the way it does

- **One responsibility per file.** `extraction.py` only extracts.
  `rag.py` only retrieves/grounds. `tools.py` only decides/acts.
  `main.py` only handles HTTP. `mcp_server.py` only exposes tools
  over MCP.
- **Schema-constrained LLM output** with explicit validation — fails
  loudly instead of silently passing malformed data downstream.
- **Rule-based agent dispatch, not an LLM loop, for tool selection.**
  The extraction step already did the reasoning (topics, mood, risk).
  Re-asking an LLM "which tool should I call" here would just add
  latency and hallucination risk for no benefit. An LLM-driven agent
  loop would be justified for a genuinely ambiguous decision — this
  isn't one.
- **`risk_flag` is a signal, not a judgment.** Grounded explicitly by
  `safety_guidelines/` — the tool never gives clinical advice or
  makes a diagnosis, only surfaces language for a human to review.
- **Single FastAPI app** serves both the HTML form and JSON API —
  one process, one deploy target.

## Safety and privacy

- All transcripts and knowledge-base content here are fabricated —
  no real client data was used anywhere in development or testing.
- The RAG index is built once at startup from a small, static
  knowledge base — nothing here is a live connection to real
  clinical records.
- `risk_flag` never triggers an automated action beyond creating a
  review item (see `knowledge/safety_guidelines/`). All clinical
  decisions remain with a human.

## Tech stack

- **LLM:** Gemini API (`gemini-2.0-flash`) — free tier, schema-constrained JSON output
- **RAG:** FAISS (flat L2 index) + Gemini embeddings (`text-embedding-004`), category-aware retrieval
- **Agent / tool-calling:** rule-based dispatcher in `tools.py`, same functions exposed via MCP
- **MCP:** `mcp` Python SDK (FastMCP)
- **Backend:** FastAPI + Uvicorn
- **Eval:** custom script (`eval.py`) comparing output to hand-written expected results

## Setup

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env   # then fill in your GEMINI_API_KEY
export GEMINI_API_KEY="your-key-here"
```

Get a free Gemini API key: https://aistudio.google.com/apikey

## Run

**Web app (FastAPI, full pipeline: RAG -> extraction -> agent):**
```bash
uvicorn main:app --reload
```
Visit `http://127.0.0.1:8000`

**Test extraction directly (no grounding):**
```bash
python extraction.py ../transcripts/session_1_normal.txt
```

**Test RAG retrieval + context building:**
```bash
python rag.py ../transcripts/session_1_normal.txt
```

**Run the agent (extraction + tool dispatch, no grounding):**
```bash
python tools.py ../transcripts/session_2_followup_heavy.txt
```

**Run the eval suite:**
```bash
python eval.py
```

**Run as an MCP server (full pipeline):**
```bash
python mcp_server.py
```

## Deploy (Render — fastest path)

1. Push this repo to GitHub.
2. On Render: New -> Web Service -> connect repo.
3. Root directory: `backend`
4. Build command: `pip install -r requirements.txt`
5. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Add environment variable: `GEMINI_API_KEY`
7. Deploy — live URL in a few minutes.

## Project structure

```
notes-copilot/
├── transcripts/                          # test input sessions + eval ground truth
│   ├── session_1_normal.txt
│   ├── session_2_followup_heavy.txt
│   ├── session_3_risk_language.txt
│   └── expected_outputs.json
├── knowledge/                            # RAG knowledge base
│   ├── documentation_guidelines/
│   │   └── note_writing_guidelines.txt
│   ├── safety_guidelines/
│   │   └── risk_flagging_protocol.txt
│   └── synthetic_previous_sessions/
│       └── session_00_prior_context.txt
├── backend/
│   ├── main.py            # FastAPI app (web form + JSON API), wires full pipeline
│   ├── extraction.py      # transcript (+ grounding context) -> structured JSON
│   ├── rag.py               # index_documents / retrieve_relevant_context / build_context
│   ├── tools.py             # agent tool dispatch
│   ├── mcp_server.py        # MCP tool exposure, wires full pipeline
│   ├── eval.py                # eval harness
│   ├── requirements.txt
│   └── .env.example
└── README.md
```

## What's next

- Frontend polish (currently a minimal built-in HTML form — a proper
  React/Next.js UI is the planned next step, kept separate so backend
  correctness didn't get blocked on UI work)
- Wire `schedule_followup_tool` to a real calendar/EMR mock integration
- Expand the eval set and knowledge base beyond this small starting point
