"""
schemas.py — Notes Copilot data schemas

Pydantic models for validating LLM output and internal data structures.
Replaces manual json.loads() + key-checking with proper schema
validation: wrong types, missing fields, and malformed structure all
fail loudly with a clear error instead of silently passing through.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List


class SessionExtraction(BaseModel):
    """
    Structured output of the extraction pipeline (extraction.py).
    This is the contract the LLM's JSON output must satisfy.
    """
    summary: str = Field(..., min_length=1, description="2-4 sentence neutral summary")
    key_topics: List[str] = Field(default_factory=list)
    mood_indicators: List[str] = Field(default_factory=list)
    follow_up_actions: List[str] = Field(default_factory=list)
    risk_flag: bool

    @field_validator("summary")
    @classmethod
    def summary_not_placeholder(cls, v: str) -> str:
        if len(v.strip()) < 10:
            raise ValueError("summary is too short to be a real extraction (possible model failure)")
        return v.strip()


class RetrievedChunk(BaseModel):
    """A single chunk retrieved by rag.py's retrieve_relevant_context()."""
    text: str
    category: str
    source: str
    distance: float


class AgentToolResult(BaseModel):
    """Generic wrapper for a single tool's output in tools.py."""
    tool: str
    status: str = ""


class NoteToolResult(AgentToolResult):
    draft_note: str


class RiskToolResult(AgentToolResult):
    flagged: bool
    reason: str = ""
    mood_indicators: List[str] = Field(default_factory=list)


class FollowupToolResult(AgentToolResult):
    session_date: str
    suggested_next_check_in: str
    actions: List[str] = Field(default_factory=list)


class AgentOutput(BaseModel):
    """Combined output of tools.run_agent()."""
    note: NoteToolResult
    risk: RiskToolResult
    followup: FollowupToolResult | None = None


class PipelineResult(BaseModel):
    """Full pipeline output — what main.py and mcp_server.py return."""
    extraction: SessionExtraction
    agent_actions: AgentOutput
    grounding_used: List[str] = Field(default_factory=list)