import os
import streamlit as st
from langchain.schema import HumanMessage, AIMessage

from pdf_qa_backend import (
    build_or_load_index,
    query_faiss_index,
    get_llm_response
)

# 1. App Setup
st.set_page_config(page_title="Conversational PDF QA Agent", layout="wide")

# 2. Load API Key from Streamlit secrets
try:
    GROQ_API_KEY = st.secrets["GROQ"]["API_KEY"]
except KeyError:
    st.error("🔑 GROQ API key not found in Streamlit secrets. Please add it under [GROQ] API_KEY.")
    st.stop()

# 3. Sidebar: PDF Upload + Debug Option + Clear Chat
st.sidebar.header("PDF & Settings")
uploaded_pdf = st.sidebar.file_uploader("Upload PDF", type=["pdf"] )
show_context = st.sidebar.checkbox("Show Retrieved Contexts")
if st.sidebar.button("Clear Conversation"):
    st.session_state.chat_history = []

# 4. Only proceed after PDF upload
if uploaded_pdf:
    os.makedirs("pdfs", exist_ok=True)
    pdf_path = os.path.join("pdfs", uploaded_pdf.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())

    # 5. Load or Build Index (cached per PDF)
    @st.cache_resource
    def load_index(path):
        return build_or_load_index(path)
    embedder, texts, index = load_index(pdf_path)

    # 6. Init chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 7. Display Title & Instructions
    st.title("💬 Conversational PDF QA Agent")
    st.markdown("Upload a PDF, ask questions, and get detailed answers with context!")

    # 8. Show Contexts if debug enabled
    def display_contexts(contexts):
        st.subheader("🔍 Retrieved Contexts")
        for i, ctx in enumerate(contexts, 1):
            st.markdown(f"**Context {i}:** {ctx}")

    # 9. Render chat history
    for msg in st.session_state.chat_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    # 10. Chat input
    if prompt := st.chat_input("Type your question..."):
        contexts = query_faiss_index(prompt, embedder, index, texts)
        if show_context:
            display_contexts(contexts)
        answer = get_llm_response(
            api_key=GROQ_API_KEY,
            query=prompt,
            contexts=contexts,
            chat_history=st.session_state.chat_history
        )
        st.session_state.chat_history.append(HumanMessage(content=prompt))
        st.session_state.chat_history.append(AIMessage(content=answer))
        with st.chat_message("assistant"):
            st.markdown(answer)
else:
    st.info("📄 Please upload a PDF file to start the conversation.")
