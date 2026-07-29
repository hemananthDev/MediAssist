"""
ingest.py
---------
Loads healthcare knowledge base documents (TXT + PDF), splits them into
chunks, embeds them using nomic-embed-text via Ollama, and stores them
in ChromaDB.

Run once before starting the chatbot:
    python ingest.py

Re-run any time you add new documents to data/ or pdfs/.
"""

import logging
import shutil
import sys
import uuid
import warnings
from pathlib import Path

# Suppress noisy deprecation and encoding warnings
warnings.filterwarnings("ignore")
logging.getLogger("pypdf").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)

import ollama
import chromadb
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
PDF_DIR         = BASE_DIR / "pdfs"
VECTORSTORE_DIR = BASE_DIR / "vectorstore" / "chroma_db"

# ── Config (must match src/rag.py) ────────────────────────────────────────────
EMBEDDING_MODEL = "nomic-embed-text"
COLLECTION_NAME = "healthcare_kb"
CHUNK_SIZE      = 800
CHUNK_OVERLAP   = 100
EMBED_BATCH     = 32


def force_delete_dir(path: Path) -> None:
    """
    Delete a directory cross-platform.
    Uses shutil.rmtree with ignore_errors so stale files don't abort the run.
    On Windows, ChromaDB file locks are released once the previous Python
    process exits — so this is safe to call at the start of a fresh run.
    """
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def load_txt_documents(data_dir: Path) -> list:
    """Load all .txt files from the data directory."""
    if not data_dir.exists():
        return []
    txt_files = list(data_dir.glob("**/*.txt"))
    if not txt_files:
        return []
    print(f"[ingest] Loading {len(txt_files)} TXT file(s) from: {data_dir}")
    loader = DirectoryLoader(
        str(data_dir),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    docs = loader.load()
    print(f"[ingest] Loaded {len(docs)} TXT document(s).")
    return docs


def load_pdf_documents(pdf_dir: Path) -> list:
    """Load all .pdf files from the pdfs directory."""
    if not pdf_dir.exists():
        return []
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    if not pdf_files:
        return []
    print(f"[ingest] Loading {len(pdf_files)} PDF file(s) from: {pdf_dir}")
    all_docs = []
    for i, pdf_path in enumerate(pdf_files, 1):
        try:
            pages = PyPDFLoader(str(pdf_path)).load()
            all_docs.extend(pages)
            print(f"[ingest]   [{i}/{len(pdf_files)}] {pdf_path.name} -> {len(pages)} page(s)")
        except Exception as e:
            print(f"[ingest]   [{i}/{len(pdf_files)}] WARNING: skipped {pdf_path.name}: {e}")
    print(f"[ingest] Loaded {len(all_docs)} PDF page(s) total.")
    return all_docs


def split_documents(docs: list) -> list:
    """Split documents into overlapping chunks for retrieval."""
    print(f"[ingest] Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    chunks = [c for c in chunks if len(c.page_content.strip()) > 50]
    print(f"[ingest] Created {len(chunks)} usable chunk(s).")
    return chunks


def build_vectorstore(chunks: list) -> None:
    """
    Embed all chunks via Ollama's native client (one HTTP request per batch),
    then write directly to ChromaDB — avoids per-chunk HTTP calls that
    exhaust Windows socket buffers.
    """
    print("[ingest] Clearing old vectorstore...")
    force_delete_dir(VECTORSTORE_DIR)
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    client     = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    texts     = [c.page_content    for c in chunks]
    metadatas = [c.metadata        for c in chunks]
    ids       = [str(uuid.uuid4()) for _ in chunks]

    total_batches = (len(chunks) + EMBED_BATCH - 1) // EMBED_BATCH
    print(f"[ingest] Embedding {len(chunks)} chunks "
          f"in {total_batches} batches of {EMBED_BATCH}...")

    for i in range(0, len(chunks), EMBED_BATCH):
        batch_num = (i // EMBED_BATCH) + 1
        b_texts   = texts[i : i + EMBED_BATCH]
        b_meta    = metadatas[i : i + EMBED_BATCH]
        b_ids     = ids[i : i + EMBED_BATCH]

        print(f"[ingest]   Batch {batch_num}/{total_batches} ({len(b_texts)} chunks)...")
        response   = ollama.embed(model=EMBEDDING_MODEL, input=b_texts)
        embeddings = response["embeddings"]

        collection.add(
            ids        = b_ids,
            embeddings = embeddings,
            documents  = b_texts,
            metadatas  = b_meta,
        )

    print(f"[ingest] Vectorstore built — {len(chunks)} chunks stored.")


def main():
    txt_docs = load_txt_documents(DATA_DIR)
    pdf_docs = load_pdf_documents(PDF_DIR)
    all_docs = txt_docs + pdf_docs

    if not all_docs:
        print("[ingest] ERROR: No documents found in data/ or pdfs/. Aborting.")
        sys.exit(1)

    print(f"\n[ingest] Total: {len(all_docs)} document(s) loaded.")
    chunks = split_documents(all_docs)
    build_vectorstore(chunks)
    print("\n[ingest] Done. Run:  streamlit run app.py")


if __name__ == "__main__":
    main()
