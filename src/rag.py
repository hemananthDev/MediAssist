"""
src/rag.py
----------
Retrieval logic: queries ChromaDB using nomic-embed-text embeddings
via the native Ollama client and returns ranked chunks with
confidence scores based on cosine distance.
"""

from pathlib import Path

import ollama
import chromadb

# ── Config ─────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "nomic-embed-text"
COLLECTION_NAME = "healthcare_kb"
TOP_K           = 4

# Cosine distance thresholds (lower distance = higher similarity)
CONFIDENCE_HIGH   = 0.25   # distance <= 0.25 → High
CONFIDENCE_MEDIUM = 0.45   # distance <= 0.45 → Medium
                            # distance >  0.45 → Low


def _distance_to_confidence(distance: float) -> str:
    """Convert a cosine distance score to a human-readable confidence label."""
    if distance <= CONFIDENCE_HIGH:
        return "High"
    elif distance <= CONFIDENCE_MEDIUM:
        return "Medium"
    else:
        return "Low"


class ChromaRetriever:
    """
    Retrieves top-k chunks from ChromaDB.
    Uses the native ollama + chromadb clients directly to avoid
    LangChain wrapper overhead and Windows socket exhaustion.
    """

    def __init__(self, vectorstore_dir: Path):
        if not vectorstore_dir.exists():
            raise FileNotFoundError(
                f"Vectorstore not found at {vectorstore_dir}. "
                "Please run `python ingest.py` first."
            )
        client = chromadb.PersistentClient(path=str(vectorstore_dir))
        self._collection = client.get_collection(COLLECTION_NAME)

    def retrieve(self, query: str, k: int = TOP_K) -> list[dict]:
        """
        Query the vectorstore and return top-k chunks.

        Each result dict contains:
            text       : str   – the chunk text
            source     : str   – source filename
            distance   : float – cosine distance (lower = more relevant)
            confidence : str   – "High" | "Medium" | "Low"
        """
        response  = ollama.embed(model=EMBEDDING_MODEL, input=[query])
        query_emb = response["embeddings"][0]

        results = self._collection.query(
            query_embeddings=[query_emb],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        docs      = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, meta, dist in zip(docs, metadatas, distances):
            chunks.append({
                "text":       doc,
                "source":     Path(meta.get("source", "")).name if meta.get("source") else "",
                "distance":   round(dist, 4),
                "confidence": _distance_to_confidence(dist),
            })

        return chunks

    def format_context(self, chunks: list[dict]) -> str:
        """Format retrieved chunks into a single context string for the prompt."""
        return "\n\n---\n\n".join(c["text"] for c in chunks)

    def top_confidence(self, chunks: list[dict]) -> str:
        """Return the confidence level of the best (closest) chunk."""
        if not chunks:
            return "Low"
        return chunks[0]["confidence"]

    def unique_sources(self, chunks: list[dict]) -> list[str]:
        """Return deduplicated list of source filenames."""
        return list(dict.fromkeys(c["source"] for c in chunks if c["source"]))
