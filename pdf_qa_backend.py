import os
import numpy as np
import faiss
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage, AIMessage

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()  # ensure GROQ_API_KEY is in env
VECTOR_DIR = "vector_store"
EMB_PATH   = os.path.join(VECTOR_DIR, "embeddings.npy")
IDX_PATH   = os.path.join(VECTOR_DIR, "faiss_index.idx")
os.makedirs(VECTOR_DIR, exist_ok=True)

EMBED_MODEL = "all-MiniLM-L6-v2"
FAISS_METRIC = faiss.METRIC_INNER_PRODUCT  # cosine after normalize

# ─────────────────────────────────────────────────────────────────────────────
# 1. PDF Loading & Chunking
# ─────────────────────────────────────────────────────────────────────────────

def load_pdf_and_chunks(pdf_path: str,
                        chunk_size: int = 500,
                        chunk_overlap: int = 50
                       ) -> list[str]:
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

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
# 2. Build or Load FAISS Index
# ─────────────────────────────────────────────────────────────────────────────

def build_or_load_index(pdf_path: str) -> tuple[SentenceTransformer, list[str], faiss.Index]:
    if os.path.exists(EMB_PATH) and os.path.exists(IDX_PATH):
        try:
            embeddings = np.load(EMB_PATH)
            index = faiss.read_index(IDX_PATH)
            embedder = SentenceTransformer(EMBED_MODEL)
            texts = load_pdf_and_chunks(pdf_path)
            return embedder, texts, index
        except Exception:
            print("⚠️ Failed to load existing index. Rebuilding...")

    texts = load_pdf_and_chunks(pdf_path)
    if not texts:
        raise ValueError("No text extracted from PDF; cannot build index.")

    embedder = SentenceTransformer(EMBED_MODEL)
    embeddings = embedder.encode(texts, show_progress_bar=True)
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlat(dim, FAISS_METRIC)
    index.add(embeddings)

    np.save(EMB_PATH, embeddings)
    faiss.write_index(index, IDX_PATH)
    return embedder, texts, index

# ─────────────────────────────────────────────────────────────────────────────
# 3. Query the FAISS Index
# ─────────────────────────────────────────────────────────────────────────────

def query_faiss_index(query: str,
                      embedder: SentenceTransformer,
                      index: faiss.Index,
                      texts: list[str],
                      k: int = 3
                     ) -> list[str]:
    if not texts:
        return []
    k = min(k, len(texts))
    q_emb = embedder.encode([query])
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    return [texts[i] for i in I[0] if 0 <= i < len(texts)]

# ─────────────────────────────────────────────────────────────────────────────
# 4. Conversational LLM Invocation
# ─────────────────────────────────────────────────────────────────────────────

def get_llm_response(api_key: str,
                     query: str,
                     contexts: list[str],
                     chat_history: list = None
                    ) -> str:
    if not api_key:
        raise ValueError("GROQ API key is required.")

    llm = ChatGroq(model_name="llama3-8b-8192", api_key=api_key)
    system_msg = SystemMessage(
        content=(
            "You are a helpful assistant. Use provided context if relevant. "
            "Maintain conversational context with the user."
        )
    )
    messages = [system_msg]
    if chat_history:
        messages.extend(chat_history)

    # Add current user question with context
    prompt = "Use the following context to answer the question.\n\n"
    for idx, ctx in enumerate(contexts, 1):
        prompt += f"Context {idx}: {ctx}\n\n"
    prompt += f"Question: {query}\nAnswer:"
    messages.append(HumanMessage(content=prompt))

    response = llm.invoke(messages)
    # Record AI response in history
    return response.content