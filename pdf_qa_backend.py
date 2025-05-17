import os
import numpy as np
import faiss
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()
VECTOR_DIR = "vector_store"
EMBED_MODEL = "all-MiniLM-L6-v2"
FAISS_METRIC = faiss.METRIC_INNER_PRODUCT  # cosine after normalize

# Ensure base vector directory exists
def _ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. PDF Loading & Chunking
# ─────────────────────────────────────────────────────────────────────────────

def load_pdf_and_chunks(pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """
    Safely load and split a PDF into text chunks. Returns [] if file missing or empty.
    """
    if not os.path.isfile(pdf_path):
        return []

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    if not pages:
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

def build_or_load_index(pdf_path: str) -> tuple[SentenceTransformer, list[str], faiss.Index]:
    """
    Builds or loads a FAISS index specific to the given PDF.
    Embedding and index files are named based on the PDF filename.
    """
    safe_name = os.path.splitext(os.path.basename(pdf_path))[0]
    emb_path = os.path.join(VECTOR_DIR, f"{safe_name}_embeddings.npy")
    idx_path = os.path.join(VECTOR_DIR, f"{safe_name}_index.idx")
    _ensure_dir(emb_path)
    _ensure_dir(idx_path)

    # Try loading existing
    if os.path.exists(emb_path) and os.path.exists(idx_path):
        try:
            embeddings = np.load(emb_path)
            index = faiss.read_index(idx_path)
            embedder = SentenceTransformer(EMBED_MODEL)
            texts = load_pdf_and_chunks(pdf_path)
            return embedder, texts, index
        except Exception:
            pass  # fallback to rebuild

    # Build from scratch
    texts = load_pdf_and_chunks(pdf_path)
    if not texts:
        # return empty index dimension 1
        embedder = SentenceTransformer(EMBED_MODEL)
        empty_index = faiss.IndexFlat(1, FAISS_METRIC)
        return embedder, [], empty_index

    embedder = SentenceTransformer(EMBED_MODEL)
    embeddings = embedder.encode(texts, show_progress_bar=True)
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlat(dim, FAISS_METRIC)
    index.add(embeddings)

    np.save(emb_path, embeddings)
    faiss.write_index(index, idx_path)
    return embedder, texts, index

# ─────────────────────────────────────────────────────────────────────────────
# 3. Query the FAISS Index
# ─────────────────────────────────────────────────────────────────────────────

def query_faiss_index(query: str, embedder, index, texts: list[str], k: int = 3) -> list[str]:
    if not texts or index.ntotal == 0:
        return []
    k = min(k, len(texts))
    q_emb = embedder.encode([query])
    faiss.normalize_L2(q_emb)
    _, I = index.search(q_emb, k)
    return [texts[i] for i in I[0] if 0 <= i < len(texts)]

# ─────────────────────────────────────────────────────────────────────────────
# 4. Conversational LLM Invocation with Markdown instruction
# ─────────────────────────────────────────────────────────────────────────────

def get_llm_response(api_key: str, query: str, contexts: list[str], chat_history: list = None) -> str:
    if not api_key:
        raise ValueError("GROQ API key is required.")

    llm = ChatGroq(model_name="llama3-8b-8192", api_key=api_key)
    system_msg = SystemMessage(
        content=(
            "You are a helpful assistant. "
            "Use provided context if relevant and format answers in Markdown. "
            "Maintain conversational context."
        )
    )
    messages = [system_msg]
    if chat_history:
        messages.extend(chat_history)

    prompt = ""
    if contexts:
        prompt = "Use the following context to answer the question in Markdown:\n\n" + \
                 "\n\n".join(f"Context {i+1}: {ctx}" for i, ctx in enumerate(contexts)) + \
                 "\n\n"
    prompt += f"Question: {query}\nAnswer:"
    messages.append(HumanMessage(content=prompt))

    response = llm.invoke(messages)
    return response.content
