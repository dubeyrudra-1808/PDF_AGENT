# Conversational PDF QA Agent

A Streamlit-powered web application that lets you upload any PDF document and interactively ask questions about its content. Under the hood, it leverages:

- **LangChain** for prompt management and conversational context handling.
- **FAISS** (via Facebook AI Similarity Search) for fast retrieval of relevant text chunks.
- **SentenceTransformers** for embedding computation (`all-MiniLM-L6-v2` model).
- **ChatGroq** (GROQ API) for LLM-powered question answering in a conversational context.

---

## 🚀 Features

- **Upload & Chat**: Simply drag-and-drop a PDF and start asking questions in natural language.
- **Context Retrieval**: Retrieves the top-𝑘 most relevant chunks from the PDF using FAISS-based vector similarity.
- **Conversational Memory**: Maintains chat history for follow-up questions.
- **Debug Mode**: Toggle “Show Retrieved Contexts” to inspect which passages informed the answers.
- **Caching**: Index and embeddings are cached per-PDF to speed up subsequent queries.
- **Lightweight & Extensible**: Minimal codebase, easy to swap models or vector backends.

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/dubeyrudra-1808/PDF_AGENT.git
   cd PDF_AGENT
   ```

2. **Create & activate a Python environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # macOS/Linux
   .\.venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   - Copy `.env.example` to `.env` (if provided) or create a `.env` file at the project root.
   - Add your GROQ API key under the `[GROQ]` section:
     ```ini
     [GROQ]
     API_KEY=your_groq_api_key_here
     ```

---

## 🏗 Architecture & Code Overview

```
PDF_AGENT/
├─ streamlit_app.py        # Main Streamlit application
├─ pdf_qa_backend.py       # Core logic: loading, chunking, indexing, querying
├─ requirements.txt        # Python dependencies
├─ .env                    # Environment variables (GROQ API key)
├─ pdfs/                   # Uploaded PDFs (created at runtime)
└─ vector_store/           # Cached embeddings and FAISS indexes
```

1. **streamlit_app.py**
   - Sets up a wide-layout Streamlit app.
   - Loads the GROQ API key from `st.secrets` or `.env`.
   - Handles PDF upload, caching of FAISS index via `@st.cache_resource`.
   - Renders chat history, debug contexts, and input box.

2. **pdf_qa_backend.py**
   - **Loading & Chunking**: Uses `PyPDFLoader` and `RecursiveCharacterTextSplitter` to break PDFs into overlapping chunks.
   - **Indexing** (`build_or_load_index`): Computes or loads embeddings (NumPy+FAISS), normalizes, and persists files under `vector_store/`.
   - **Querying** (`query_faiss_index`): Encodes user queries, finds top-𝑘 similar chunks.
   - **LLM Invocation** (`get_llm_response`): Constructs a system prompt, appends chat history, injects retrieved contexts, and calls GROQ’s `ChatGroq`.

---

## 🎯 Usage

1. **Run the app**
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Open in browser**
   - By default: http://localhost:8501

3. **Interact**
   - Upload your PDF.
   - Ask questions in the chat interface.
   - Optionally toggle **Show Retrieved Contexts** to debug.

---

## 🔧 Configuration

- **Embedding Model**: Change `EMBED_MODEL` (`all-MiniLM-L6-v2`) in `pdf_qa_backend.py` to another `sentence-transformers` model.
- **FAISS Metric**: Adjust `FAISS_METRIC` for inner-product or L2.
- **Chunk Size & Overlap**: Tune `chunk_size` and `chunk_overlap` in `load_pdf_and_chunks`.

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or pull requests for bug fixes, enhancements, or new features.

1. Fork the repo.
2. Create a feature branch: `git checkout -b feat/my-new-feature`.
3. Commit your changes: `git commit -m "feat: add awesome feature"`.
4. Push to the branch: `git push origin feat/my-new-feature`.
5. Open a pull request.

---

## 📝 License

This project is licensed under the MIT License. See `LICENSE` for more details.

---

_Developed with ❤️ by dubeyrudra-1808._

