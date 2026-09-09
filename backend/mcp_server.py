"""
mcp_server.py — Notes Copilot MCP server

Exposes the full pipeline (RAG grounding + extraction + agent tools)
via the Model Context Protocol, so any MCP client (Claude Desktop,
Claude Code, etc.) can call it directly without a custom integration.

This is a thin wrapper — all real logic lives in rag.py, extraction.py,
and tools.py. This file just declares the MCP tool interface.

Run:
    python mcp_server.py
"""

import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

try:
    from mcp.server.mcpserver import MCPServer
    mcp = MCPServer("notes-copilot")
except (ImportError, ModuleNotFoundError):
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("notes-copilot")

KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "..", "knowledge")
_rag_index = None
_rag_chunks = None


def _get_rag_index():
    global _rag_index, _rag_chunks
    if _rag_index is None:
        _rag_index, _rag_chunks = index_documents(KNOWLEDGE_DIR)
    return _rag_index, _rag_chunks


@mcp.tool()
def summarize_session(transcript: str) -> dict:
    """
    Extract a structured summary (summary, key topics, mood indicators,
    follow-up actions, risk flag) from a therapy session transcript,
    grounded with relevant documentation guidelines, safety guidelines,
    and prior session context retrieved via RAG.
    """
    index, chunks = _get_rag_index()
    relevant = retrieve_relevant_context(index, chunks, transcript, k=3)
    context = build_context(relevant)
    return extract_session_notes(transcript, context=context)


@mcp.tool()
def process_session(transcript: str, session_name: str = "unnamed_session") -> dict:
    """
    Full pipeline: retrieves grounding context, extracts structured
    notes, then runs the agent to generate a draft note, flag risk if
    needed, and prepare follow-up scheduling.
    """
    index, chunks = _get_rag_index()
    relevant = retrieve_relevant_context(index, chunks, transcript, k=3)
    context = build_context(relevant)

    extraction = extract_session_notes(transcript, context=context)
    agent_output = run_agent(extraction, session_name=session_name)

    return {
        "extraction": extraction,
        "agent_actions": agent_output,
        "grounding_used": [r["source"] for r in relevant],
    }


if __name__ == "__main__":
    mcp.run()
