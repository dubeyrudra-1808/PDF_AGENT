import os
from dotenv import load_dotenv
import streamlit as st
from langchain.schema import HumanMessage, AIMessage

from pdf_qa_backend import (
    build_or_load_index,
    query_faiss_index,
    get_llm_response
)

# 1. App Setup
st.set_page_config(page_title="Conversational PDF QA Agent", layout="wide")
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("🔑 GROQ_API_KEY not found in .env. Please add it.")
    st.stop()

# 2. Sidebar: PDF Upload
st.sidebar.header("📁 Upload PDF")
uploaded_pdf = st.sidebar.file_uploader("Choose a PDF", type=["pdf"])
if uploaded_pdf:
    os.makedirs("pdfs", exist_ok=True)
    pdf_path = os.path.join("pdfs", uploaded_pdf.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())
else:
    pdf_path = "pdfs/fine_tuning.pdf"

# 3. Load or Build Index
with st.spinner("🔄 Loading or building index..."):
    embedder, texts, index = build_or_load_index(pdf_path)

# 4. Initialize Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 5. Display Title
st.title("💬 Conversational PDF QA Agent")
st.markdown("Ask questions about your PDF — your chat history will be remembered.")

# 6. Render Chat Messages
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# 7. Chat Input at Bottom
if user_input := st.chat_input("Type your question..."):
    # Retrieve relevant contexts
    contexts = query_faiss_index(user_input, embedder, index, texts)
    # Get LLM response with history
    answer = get_llm_response(
        api_key=GROQ_API_KEY,
        query=user_input,
        contexts=contexts,
        chat_history=st.session_state.chat_history
    )
    # Store messages
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=answer))
    # Display latest exchange immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        st.markdown(answer)
