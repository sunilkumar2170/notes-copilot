"""
rag.py — Notes Copilot RAG layer

Grounds extraction in real reference material instead of retrieving
"vaguely similar text." The knowledge base has three distinct
categories, each with a different purpose:

  knowledge/documentation_guidelines/   -> how a clinical note should
                                            be structured (grounds the
                                            summary/note quality)
  knowledge/safety_guidelines/          -> how risk-flagged content
                                            should be handled (grounds
                                            risk_flag decisions and
                                            what the tool must NOT do)
  knowledge/synthetic_previous_sessions/ -> prior sessions for the same
                                            fictional client (grounds
                                            recurring-theme detection)

Three functions, each with one job:
  index_documents()           -> build a FAISS index over the knowledge base
  retrieve_relevant_context() -> get top-k relevant chunks for a query
  build_context()              -> format retrieved chunks into a prompt-ready string

embed_text() is exposed (not underscore-prefixed) because eval.py also
uses it for semantic similarity scoring — it's shared infrastructure,
not RAG-internal only.

Uses Gemini's embedding model (free tier) — no separate embedding
service needed, keeps the stack small.
"""

import os
import glob
import numpy as np
import faiss
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
EMBEDDING_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

try:
    from google import genai
    USE_MODERN_SDK = True
except ImportError:
    import google.generativeai as legacy_genai
    USE_MODERN_SDK = False

KNOWLEDGE_CATEGORIES = {
    "documentation_guidelines": "documentation_guidelines",
    "safety_guidelines": "safety_guidelines",
    "synthetic_previous_sessions": "synthetic_previous_sessions",
}


_EMBED_CACHE = {}


def embed_text(text: str) -> np.ndarray:
    """Get an embedding vector for a piece of text via Gemini.
    Shared by rag.py (indexing/retrieval) and eval.py (similarity scoring)."""
    text_key = text.strip()
    if text_key in _EMBED_CACHE:
        return _EMBED_CACHE[text_key]

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY or GOOGLE_API_KEY environment variable is not set. "
            "Please set it in your environment or a .env file."
        )

    if USE_MODERN_SDK:
        client = genai.Client(api_key=api_key)
        for model_candidate in [EMBEDDING_MODEL, "gemini-embedding-001", "models/gemini-embedding-001", "gemini-embedding-2-preview"]:
            try:
                result = client.models.embed_content(
                    model=model_candidate,
                    contents=text,
                )
                if hasattr(result, "embeddings") and result.embeddings:
                    emb = np.array(result.embeddings[0].values, dtype="float32")
                elif hasattr(result, "embedding") and hasattr(result.embedding, "values"):
                    emb = np.array(result.embedding.values, dtype="float32")
                elif hasattr(result, "embedding"):
                    emb = np.array(result.embedding, dtype="float32")
                else:
                    emb = np.array(result, dtype="float32")
                _EMBED_CACHE[text_key] = emb
                return emb
            except Exception:
                continue
        raise ValueError("Embedding generation failed across all available Gemini embedding models.")
    else:
        legacy_genai.configure(api_key=api_key)
        for model_candidate in [EMBEDDING_MODEL, "models/gemini-embedding-001", "models/text-embedding-004"]:
            try:
                result = legacy_genai.embed_content(model=model_candidate, content=text)
                emb = np.array(result["embedding"], dtype="float32")
                _EMBED_CACHE[text_key] = emb
                return emb
            except Exception:
                continue
        raise ValueError("Legacy embedding generation failed across all available Gemini embedding models.")


def _chunk_text(text: str, chunk_size: int = 400) -> list:
    """Simple fixed-size word chunking — documents here are short
    enough that this is sufficient without a fancier splitter."""
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)] or [text]


def index_documents(knowledge_dir: str):
    """
    Walks knowledge_dir's three category subfolders, chunks every .txt
    file, embeds each chunk, and builds a single FAISS flat L2 index
    over all of them. Each chunk record carries its category and
    source filename so retrieval results are traceable and filterable.

    Returns (index, chunk_records).
    """
    chunk_records = []  # each: {"text", "category", "source"}

    for category in KNOWLEDGE_CATEGORIES.values():
        category_dir = os.path.join(knowledge_dir, category)
        if not os.path.isdir(category_dir):
            continue
        for filepath in sorted(glob.glob(os.path.join(category_dir, "*.txt"))):
            with open(filepath, "r") as f:
                text = f.read()
            for chunk in _chunk_text(text):
                chunk_records.append({
                    "text": chunk,
                    "category": category,
                    "source": os.path.basename(filepath),
                })

    if not chunk_records:
        raise ValueError(f"No knowledge documents found under {knowledge_dir}")

    embeddings = np.array([embed_text(r["text"]) for r in chunk_records], dtype="float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, chunk_records


def retrieve_relevant_context(index, chunk_records: list, query_text: str, k: int = 3, category_filter: str = None) -> list:
    """
    Embeds query_text and returns the top-k most similar chunks.
    If category_filter is set (e.g. "safety_guidelines"), only
    considers chunks from that category.
    """
    query_embedding = embed_text(query_text).reshape(1, -1)

    search_k = min(len(chunk_records), k * 4 if category_filter else k)
    distances, indices = index.search(query_embedding, search_k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx == -1:
            continue
        record = chunk_records[idx]
        if category_filter and record["category"] != category_filter:
            continue
        results.append({
            "text": record["text"],
            "category": record["category"],
            "source": record["source"],
            "distance": float(dist),
        })
        if len(results) >= k:
            break

    return results


def build_context(retrieved_chunks: list) -> str:
    """
    Formats retrieved chunks into a single prompt-ready string,
    grouped by category so the extraction prompt can clearly
    distinguish "here is a documentation rule" from "here is prior
    session history" from "here is a safety protocol."
    """
    if not retrieved_chunks:
        return ""

    grouped = {}
    for chunk in retrieved_chunks:
        grouped.setdefault(chunk["category"], []).append(chunk["text"])

    labels = {
        "documentation_guidelines": "Documentation guidelines",
        "safety_guidelines": "Safety guidelines",
        "synthetic_previous_sessions": "Relevant prior session context",
    }

    sections = []
    for category, texts in grouped.items():
        label = labels.get(category, category)
        joined = "\n---\n".join(texts)
        sections.append(f"[{label}]\n{joined}")

    return "\n\n".join(sections)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python rag.py <path_to_new_transcript.txt>")
        sys.exit(1)

    knowledge_dir = os.path.join(os.path.dirname(__file__), "..", "knowledge")
    index, chunk_records = index_documents(knowledge_dir)

    with open(sys.argv[1], "r") as f:
        query_transcript = f.read()

    relevant = retrieve_relevant_context(index, chunk_records, query_transcript, k=3)
    context_str = build_context(relevant)

    print(context_str)