
import os
import hashlib
import tempfile

import streamlit as st
import chromadb

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Modern LangChain installations may provide chains through langchain_classic.
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

load_dotenv()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="PDF Chat AI",
    page_icon="📚",
    layout="wide",
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0b1020, #111827 65%, #172554);
        color: #f1f5f9;
    }

    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        background: linear-gradient(120deg, #172554, #164e63);
        border: 1px solid #334155;
        border-radius: 22px;
        padding: 28px;
        margin-bottom: 22px;
    }

    .hero h1 {
        color: #ffffff;
        font-size: 35px;
        margin: 0 0 8px 0;
    }

    .hero p {
        color: #cbd5e1;
        font-size: 16px;
        margin: 0;
    }

    .panel {
        background: rgba(30, 41, 59, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.23);
        border-radius: 16px;
        padding: 18px;
        margin-bottom: 16px;
    }

    div.stButton > button {
        border: 0;
        border-radius: 10px;
        color: white;
        font-weight: 600;
        background: linear-gradient(90deg, #0891b2, #2563eb);
        min-height: 42px;
    }

    div.stButton > button:hover {
        background: linear-gradient(90deg, #0e7490, #1d4ed8);
        color: white;
    }

    div[data-testid="stChatMessage"] {
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 14px;
        padding: 12px;
        background: rgba(30, 41, 59, 0.45);
    }

    div[data-testid="stTextInput"] input {
        border-radius: 10px;
    }

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 13px;
        padding-top: 22px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
defaults = {
    "store": {},
    "vectors": None,
    "document_signature": None,
    "document_names": [],
    "document_count": 0,
    "chunk_count": 0,
    "active_signature": None,
    "active_session": "default_session",
    "messages": [],
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>📚 Conversational RAG Assistant</h1>
    <p>Upload your PDFs, ask questions, and explore their content
    through an AI assistant that remembers your conversation.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    api_key = st.text_input(
        "Groq API key",
        type="password",
        value="",
        help="Enter your key for this session. Do not share it.",
    )

    session_id = st.text_input(
        "Conversation ID",
        value=st.session_state.active_session,
    ).strip()

    if not session_id:
        session_id = "default_session"

    if session_id != st.session_state.active_session:
        st.session_state.active_session = session_id
        st.session_state.messages = []

    st.divider()
    st.markdown("### 🧠 AI Pipeline")
    st.caption("LLM: Groq")
    st.caption("Embeddings: HuggingFace")
    st.caption("Vector store: ChromaDB")
    st.caption("Retrieval: History-aware RAG")

    if st.session_state.vectors is not None:
        st.success("PDF index ready")
        st.caption(f"PDFs: {st.session_state.document_count}")
        st.caption(f"Text chunks: {st.session_state.chunk_count}")
    else:
        st.info("Upload and process PDFs to begin.")

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        if session_id in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        st.rerun()

# ---------------- UPLOAD AREA ----------------
st.markdown("### 📂 Your knowledge base")

uploaded_files = st.file_uploader(
    "Upload one or more PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload research papers, reports, or other PDFs.",
)

# Use the uploaded content itself to identify the current document set.
signature = None
if uploaded_files:
    digest = hashlib.sha256()
    for uploaded_file in uploaded_files:
        digest.update(uploaded_file.name.encode("utf-8"))
        digest.update(uploaded_file.getvalue())
    signature = digest.hexdigest()

# ---------------- BUILD VECTOR STORE ----------------
if uploaded_files:
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(
            f"**{len(uploaded_files)} PDF file(s) selected**"
        )
        for uploaded_file in uploaded_files:
            st.caption(f"📄 {uploaded_file.name}")

    with col2:
        process_clicked = st.button(
            "🚀 Process PDFs",
            use_container_width=True,
        )

    if process_clicked:
        try:
            with st.spinner(
                "Reading PDFs, splitting text, and creating embeddings..."
            ):
                documents = []

                # Each PDF gets its own temporary file so files
                # do not overwrite one another.
                for uploaded_file in uploaded_files:
                    temp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(
                            suffix=".pdf", delete=False
                        ) as temp_file:
                            temp_file.write(uploaded_file.getvalue())
                            temp_path = temp_file.name

                        loader = PyPDFLoader(temp_path)
                        pdf_docs = loader.load()

                        for doc in pdf_docs:
                            doc.metadata["source"] = uploaded_file.name

                        documents.extend(pdf_docs)

                    finally:
                        if temp_path and os.path.exists(temp_path):
                            os.remove(temp_path)

                if not documents:
                    st.error(
                        "No readable text was found in the uploaded PDFs."
                    )
                else:
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=150,
                    )
                    splits = splitter.split_documents(documents)

                    if not splits:
                        st.error("No text chunks could be created.")
                    else:
                        embeddings = HuggingFaceEmbeddings(
                            model_name="sentence-transformers/all-MiniLM-L6-v2"
                        )

                        # Explicit in-memory Chroma client. No tenant
                        # or persistent database configuration is needed.
                        chroma_client = chromadb.EphemeralClient()

                        vectorstore = Chroma.from_documents(
                            documents=splits,
                            embedding=embeddings,
                            client=chroma_client,
                            collection_name="pdf_documents",
                        )

                        st.session_state.vectors = vectorstore
                        st.session_state.document_signature = signature
                        st.session_state.document_names = [
                            file.name for file in uploaded_files
                        ]
                        st.session_state.document_count = len(uploaded_files)
                        st.session_state.chunk_count = len(splits)
                        st.session_state.active_signature = signature
                        st.session_state.messages = []

            if st.session_state.vectors is not None:
                st.success("PDFs processed successfully. Start chatting below!")

        except Exception as error:
            st.error(f"Could not process PDFs: {error}")

# ---------------- CHAT SETUP ----------------
def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]


def build_rag_chain(api_key, vectorstore):
    llm = ChatGroq(
        api_key=api_key,
        model="openai/gpt-oss-20b",
        temperature=0,
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    contextualize_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Given the chat history and the latest user question, "
            "rewrite the question as a standalone question if needed. "
            "Do not answer it. If no rewrite is needed, return it unchanged.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_prompt,
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You answer questions using the retrieved PDF context only. "
            "If the answer is not in the context, say you do not know. "
            "Keep the answer concise and do not invent facts.\n\n"
            "Context:\n{context}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    document_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        document_chain,
    )

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )


# ---------------- CHAT UI ----------------
st.markdown("### 💬 Chat with your PDFs")

if st.session_state.vectors is None:
    st.info("Upload PDFs and click **Process PDFs** before asking questions.")
elif not api_key:
    st.warning("Enter your Groq API key in the sidebar to start chatting.")
else:
    # Keep the same chain for the current in-memory vector store and key.
    chain_signature = (
        st.session_state.document_signature,
        hashlib.sha256(api_key.encode()).hexdigest(),
    )

    if st.session_state.get("chain_signature") != chain_signature:
        st.session_state.conversational_chain = build_rag_chain(
            api_key,
            st.session_state.vectors,
        )
        st.session_state.chain_signature = chain_signature

    history = get_session_history(session_id)

    # Render existing conversation.
    for message in history.messages:
        if message.type == "human":
            with st.chat_message("user"):
                st.markdown(message.content)
        elif message.type == "ai":
            with st.chat_message("assistant"):
                st.markdown(message.content)

    question = st.chat_input("Ask a question about your PDFs...")

    if question:
        with st.chat_message("user"):
            st.markdown(question)

        try:
            with st.chat_message("assistant"):
                with st.spinner("Searching your PDFs and preparing an answer..."):
                    response = st.session_state.conversational_chain.invoke(
                        {"input": question},
                        config={
                            "configurable": {"session_id": session_id}
                        },
                    )

                st.markdown(response["answer"])

                with st.expander("📖 View retrieved passages"):
                    for index, doc in enumerate(
                        response.get("context", []), start=1
                    ):
                        st.markdown(f"**Passage {index}**")
                        st.caption(
                            f"Source: {doc.metadata.get('source', 'PDF document')}"
                        )
                        st.write(doc.page_content)
                        st.divider()

            # Streamlit reruns after each interaction; chat history
            # is persisted in session_state through RunnableWithMessageHistory.
            st.rerun()

        except Exception as error:
            st.error(f"Something went wrong: {error}")

st.markdown("""
<div class="footer">
    Conversational RAG Assistant · Streamlit · LangChain · ChromaDB · Groq
</div>
""", unsafe_allow_html=True)
