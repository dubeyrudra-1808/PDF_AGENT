#!/usr/bin/env python3
import os
import time
import traceback
from dotenv import load_dotenv

import numpy as np
import faiss

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

# ─────────────────────────────────────────────────────────────────────────────
# 1. Configuration & paths
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()  # load GROQ_API_KEY and optional GROQ_MODEL from .env

PDF_PATH    = "pdfs/fine_tuning.pdf"
VECTOR_DIR  = "vector_store"
EMB_PATH    = os.path.join(VECTOR_DIR, "embeddings.npy")
IDX_PATH    = os.path.join(VECTOR_DIR, "faiss_index.idx")

EMBED_MODEL = "all-MiniLM-L6-v2"

# Default production-safe model, overridable by environment
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
MODEL_FALLBACKS = [
    DEFAULT_GROQ_MODEL,
    "llama-3-groq-8B-Tool-Use",
    "llama-3-groq-70B-Tool-Use",
]

os.makedirs(VECTOR_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Prepare or load embeddings + FAISS index
# ─────────────────────────────────────────────────────────────────────────────

need_build = True
embedder = None
texts = []
index = None
embeddings = None

if os.path.exists(EMB_PATH) and os.path.exists(IDX_PATH):
    try:
        print("🔄 Loading embeddings & FAISS index from disk...")
        # load embeddings and cast to float32 for FAISS
        embeddings = np.load(EMB_PATH).astype("float32")
        index      = faiss.read_index(IDX_PATH)
        embedder   = SentenceTransformer(EMBED_MODEL)
        # load texts for retrieval
        loader   = PyPDFLoader(PDF_PATH)
        pages    = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size    = 500,
            chunk_overlap = 50,
            separators    = ["\n\n", "\n", " ", ""]
        )
        chunks   = splitter.split_documents(pages)
        texts    = [chunk.page_content for chunk in chunks]
        need_build = False
        print("✅ Loaded saved embeddings & index.")
    except Exception as e:
        print("⚠️ Failed to load saved index:", e)
        print("→ Will rebuild embeddings & index.")

if need_build:
    print("🚧 Building embeddings & FAISS index (this runs once)...")

    # 2b.1 Load & split PDF
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF not found at '{PDF_PATH}'")

    loader   = PyPDFLoader(PDF_PATH)
    pages    = loader.load()
    if not pages:
        raise RuntimeError("PDF loaded but contains no pages.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = 500,
        chunk_overlap = 50,
        separators    = ["\n\n", "\n", " ", ""]
    )
    chunks   = splitter.split_documents(pages)
    texts    = [chunk.page_content for chunk in chunks]

    # 2b.2 Embed all chunks
    embedder   = SentenceTransformer(EMBED_MODEL)
    embeddings = embedder.encode(texts, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype="float32")  # required for FAISS

    # 2b.3 Build FAISS index (cosine via inner product on normalized vectors)
    faiss.normalize_L2(embeddings)                       # normalize first
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # 2b.4 Save to disk
    np.save(EMB_PATH, embeddings)
    faiss.write_index(index, IDX_PATH)
    print(f"✅ Saved embeddings → {EMB_PATH}")
    print(f"✅ Saved FAISS index → {IDX_PATH}")

# Safety check: ensure embedder/index are present
if embedder is None or index is None:
    raise RuntimeError("Embedder or FAISS index wasn't initialized properly.")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Initialize Groq LLM client (with fallback)
# ─────────────────────────────────────────────────────────────────────────────

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    # if you run this in Streamlit, prefer to read st.secrets instead
    try:
        import streamlit as st  # type: ignore
        groq_api_key = st.secrets["GROQ"]["API_KEY"]
    except Exception:
        raise ValueError("🔑 Please set your GROQ_API_KEY in .env or Streamlit secrets.")

llm = None
last_err = None
for candidate in MODEL_FALLBACKS:
    try:
        print(f"→ Trying Groq model: {candidate}")
        llm = ChatGroq(model_name=candidate, api_key=groq_api_key)
        print(f"✅ Initialized Groq model: {candidate}")
        break
    except Exception as e:
        last_err = e
        print(f"⚠️ Failed to initialize model {candidate}: {e}")
        time.sleep(0.5)

if llm is None:
    print("❌ Could not initialize any Groq model.")
    traceback.print_exception(type(last_err), last_err, last_err.__traceback__)
    raise RuntimeError("No available Groq models. Update GROQ_MODEL or check your API key/account.")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Interactive query loop
# ─────────────────────────────────────────────────────────────────────────────

print("\n🤖 PDF QA Agent ready! Type 'exit' to quit.\n")

while True:
    try:
        query = input("📝 Your question: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n👋 Goodbye!")
        break

    if query.lower() in ("exit", "quit"):
        print("👋 Goodbye!")
        break
    if not query:
        continue

    # 4a) Embed the query and search FAISS
    try:
        q_emb = embedder.encode([query])
        q_emb = np.asarray(q_emb, dtype="float32")
        faiss.normalize_L2(q_emb)
        # handle case index has 0 vectors
        if index.ntotal == 0:
            contexts = []
        else:
            D, I = index.search(q_emb, k=min(3, index.ntotal))
            contexts = [texts[i] for i in I[0] if 0 <= i < len(texts)]
    except Exception as e:
        print("⚠️ Error during FAISS search:", e)
        traceback.print_exception(type(e), e, e.__traceback__)
        contexts = []

    # 4c) Build chat messages
    system_msg = SystemMessage(
        content=(
            "You are a helpful assistant. "
            "If the user’s question relates to the provided context, use that context. "
            "Otherwise, answer generally."
        )
    )

    user_content = "Use the following extracted context to answer the question.\n\n"
    for idx, ctx in enumerate(contexts, 1):
        user_content += f"Context {idx}:\n{ctx}\n\n"
    user_content += f"Question: {query}\nAnswer:"
    user_msg = HumanMessage(content=user_content)

    # 4d) Call Groq LLM with .invoke()
    try:
        response = llm.invoke([system_msg, user_msg])
    except Exception as e:
        print("⚠️ LLM invocation failed:", e)
        traceback.print_exception(type(e), e, e.__traceback__)
        print("Try setting GROQ_MODEL in .env to a supported model like 'llama-3.3-70b-versatile'.")
        continue

    # 4e) Display the result
    print("\n💡 Answer:\n")
    print(response.content)
    print("\n" + "—" * 40 + "\n")
