#!/usr/bin/env python3
import os
import time
import traceback
from dotenv import load_dotenv
from typing import List, Tuple, Optional

import numpy as np
import faiss

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()

PDF_PATH = "pdfs/fine_tuning.pdf"
VECTOR_DIR = "vector_store"
EMBED_MODEL = "all-MiniLM-L6-v2"
# We'll use Inner Product on normalized vectors to get cosine similarity
# Use IndexFlatIP(dim) for inner product.
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_FALLBACKS = [
    DEFAULT_GROQ_MODEL,
    "llama-3-groq-8B-Tool-Use",
    "llama-3-groq-70B-Tool-Use",
]

os.makedirs(VECTOR_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. PDF Loading & Chunking
# ─────────────────────────────────────────────────────────────────────────────

def load_pdf_and_chunks(pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """
    Load the PDF and split into text chunks. Return list of chunk strings.
    """
    if not os.path.isfile(pdf_path):
        print(f"⚠️ PDF not found: {pdf_path}")
        return []

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    if not pages:
        print("⚠️ No pages loaded from PDF.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = splitter.split_documents(pages)
    return [doc.page_content for doc in docs]

# ─────────────────────────────────────────────────────────────────────────────
# 2. Build or Load FAISS Index (per-PDF)
# ─────────────────────────────────────────────────────────────────────────────

def build_or_load_index(pdf_path: str) -> Tuple[SentenceTransformer, List[str], faiss.Index]:
    """
    Builds or loads a FAISS index specific to the given PDF.
    Embedding and index files are named based on the PDF filename.
    """
    safe_name = os.path.splitext(os.path.basename(pdf_path))[0]
    emb_path = os.path.join(VECTOR_DIR, f"{safe_name}_embeddings.npy")
    idx_path = os.path.join(VECTOR_DIR, f"{safe_name}_index.idx")
    ensure_parent_dir(emb_path)
    ensure_parent_dir(idx_path)

    # Try loading existing
    if os.path.exists(emb_path) and os.path.exists(idx_path):
        try:
            print("🔄 Loading saved embeddings & FAISS index from disk...")
            embeddings = np.load(emb_path)
            # Ensure float32 for FAISS
            embeddings = embeddings.astype("float32")
            index = faiss.read_index(idx_path)
            embedder = SentenceTransformer(EMBED_MODEL)
            texts = load_pdf_and_chunks(pdf_path)
            return embedder, texts, index
        except Exception as e:
            print("⚠️ Failed to load saved index (will rebuild):", e)

    # Build from scratch
    texts = load_pdf_and_chunks(pdf_path)
    if not texts:
        print("ℹ️ No text chunks found; returning empty index.")
        embedder = SentenceTransformer(EMBED_MODEL)
        empty_index = faiss.IndexFlatIP(1)  # dimension placeholder
        return embedder, [], empty_index

    embedder = SentenceTransformer(EMBED_MODEL)
    print(f"🚧 Generating embeddings for {len(texts)} chunks (this may take a bit)...")
    embeddings = embedder.encode(texts, show_progress_bar=True)
    # Ensure correct dtype for FAISS
    embeddings = np.asarray(embeddings, dtype="float32")

    # Normalize to unit length so inner product == cosine similarity
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Persist to disk
    np.save(emb_path, embeddings)
    faiss.write_index(index, idx_path)
    print(f"✅ Saved embeddings → {emb_path}")
    print(f"✅ Saved FAISS index → {idx_path}")

    return embedder, texts, index

# ─────────────────────────────────────────────────────────────────────────────
# 3. Query the FAISS Index
# ─────────────────────────────────────────────────────────────────────────────

def query_faiss_index(query: str, embedder: SentenceTransformer, index: faiss.Index, texts: List[str], k: int = 3) -> List[str]:
    if not texts or index.ntotal == 0:
        return []
    k = min(k, len(texts))
    q_emb = embedder.encode([query])
    q_emb = np.asarray(q_emb, dtype="float32")
    faiss.normalize_L2(q_emb)
    _, I = index.search(q_emb, k)
    return [texts[i] for i in I[0] if 0 <= i < len(texts)]

# ─────────────────────────────────────────────────────────────────────────────
# 4. Groq LLM init with fallbacks
# ─────────────────────────────────────────────────────────────────────────────

def initialize_groq_client(api_key: str, fallbacks: List[str]) -> ChatGroq:
    if not api_key:
        raise ValueError("GROQ_API_KEY is required (set it in your .env).")

    last_err = None
    for candidate in fallbacks:
        try:
            print(f"→ Trying Groq model: {candidate}")
            client = ChatGroq(model_name=candidate, api_key=api_key)
            # If constructor succeeds, assume usable
            print(f"✅ Initialized Groq model: {candidate}")
            return client
        except Exception as e:
            last_err = e
            print(f"⚠️ Failed to initialize {candidate}: {e}")
            time.sleep(0.3)

    print("❌ Unable to initialize any Groq model from fallbacks.")
    if last_err:
        traceback.print_exception(type(last_err), last_err, last_err.__traceback__)
    raise RuntimeError("No available Groq models. Update GROQ_MODEL or check your GROQ account and API key.")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Build prompt and call LLM
# ─────────────────────────────────────────────────────────────────────────────

def get_llm_response(llm: ChatGroq, query: str, contexts: List[str], chat_history: Optional[List] = None) -> str:
    system_msg = SystemMessage(
        content=(
            "You are a helpful assistant. "
            "Use provided context if relevant and format answers in Markdown. "
            "If the context doesn't contain the answer, respond concisely and clearly."
        )
    )

    messages = [system_msg]
    if chat_history:
        messages.extend(chat_history)

    prompt = ""
    if contexts:
        prompt = "Use the following context (from document) to answer the question in Markdown:\n\n"
        for i, ctx in enumerate(contexts):
            prompt += f"Context {i+1}:\n{ctx}\n\n"
    prompt += f"Question: {query}\nAnswer:"
    messages.append(HumanMessage(content=prompt))

    try:
        response = llm.invoke(messages)
    except Exception as e:
        print("⚠️ LLM invocation failed:", e)
        traceback.print_exception(type(e), e, e.__traceback__)
        raise

    return response.content

# ─────────────────────────────────────────────────────────────────────────────
# Main interactive agent
# ─────────────────────────────────────────────────────────────────────────────

def main():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY missing. Add it to your .env file.")

    # Initialize Groq client (tries fallbacks)
    llm = initialize_groq_client(groq_api_key, GROQ_FALLBACKS)

    # Prepare index for the target PDF
    embedder, texts, index = build_or_load_index(PDF_PATH)

    print("\n🤖 PDF QA Agent ready! Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            query = input("📝 Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            print("👋 Goodbye!")
            break

        try:
            contexts = query_faiss_index(query, embedder, index, texts, k=3)
        except Exception as e:
            print("⚠️ Error during FAISS search:", e)
            contexts = []

        try:
            answer = get_llm_response(llm, query, contexts)
        except Exception:
            print("⚠️ LLM failed to produce an answer. Consider checking GROQ_MODEL or your API key.")
            continue

        print("\n💡 Answer (Markdown):\n")
        print(answer)
        print("\n" + "-" * 60 + "\n")

if __name__ == "__main__":
    main()
