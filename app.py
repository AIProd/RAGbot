import os
import tempfile
import textwrap
from pathlib import Path
from typing import List

import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# ──────────────────────────────────────────────────────────────────────────────
# 1. Secrets / Environment configuration
# ──────────────────────────────────────────────────────────────────────────────
# Make sure you add AZURE_ENDPOINT, AZURE_API_KEY and (optionally) API_VERSION
# to Streamlit ➜ Settings ➜ Secrets when deploying on Streamlit Community Cloud.
# They will be available through st.secrets; we fall back to environment vars
# when running locally.
# ──────────────────────────────────────────────────────────────────────────────
AZURE_ENDPOINT = st.secrets.get("AZURE_ENDPOINT", os.getenv("AZURE_ENDPOINT"))
AZURE_API_KEY = st.secrets.get("AZURE_API_KEY", os.getenv("AZURE_API_KEY"))
API_VERSION = st.secrets.get("API_VERSION", os.getenv("API_VERSION", "2024-12-01-preview"))

# Model / deployment names on your Azure OpenAI resource ----------------------
CHAT_DEPLOYMENT = "gpt-4o"                   # name of chat deployment
aEMBED_MODEL   = "text-embedding-3-large"    # name of embedding deployment

# -----------------------------------------------------------------------------
# 2. Utility helpers
# -----------------------------------------------------------------------------

def _write_uploaded_file(uploaded_file) -> Path:
    """Write Streamlit UploadedFile to a temporary path and return it."""
    suffix = Path(uploaded_file.name).suffix
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(tmp_fd, "wb") as f:
        f.write(uploaded_file.read())
    return Path(tmp_path)


def _build_index(pdf_files: List[Path]):
    """Given a list of PDF file paths, build (or rebuild) the FAISS index
    and LangChain RAG pipeline, saving index to ./faiss_index."""

    # 2·1  Load & split -------------------------------------------------------
    docs = []
    for p in pdf_files:
        docs.extend(PyPDFLoader(str(p)).load_and_split())

    splitter = RecursiveCharacterTextSplitter(chunk_size=3200, chunk_overlap=800)
    splits = splitter.split_documents(docs)

    # 2·2  Embed & store ------------------------------------------------------
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        model=aEMBED_MODEL,
        chunk_size=2048,
    )

    vectordb = FAISS.from_documents(splits, embeddings)

    # Persist so we can reuse without recomputing every reload
    vectordb.save_local("faiss_index")

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    # 2·3  Build RAG chain ----------------------------------------------------
    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        openai_api_version=API_VERSION,
        azure_deployment=CHAT_DEPLOYMENT,
        temperature=0,
        max_tokens=4096,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True,
    )

    prompt_template = (
        "You are a helpful assistant for BobiHealth.\n\n"
        "Use the following context to answer the user's question.\n"
        "If you don't know the answer, say you don't know—don't fabricate.\n"
        "Response should not be in more than 30‑50 words.\n\n"
        "{context}\n\n"
        "Chat History:\n{chat_history}\n\n"
        "User question: {question}\n"
        "Helpful answer:"
    )

    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=prompt_template,
    )

    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        output_key="answer",
    )

    return rag_chain, vectordb


# -----------------------------------------------------------------------------
# 3. Streamlit Page config
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="BobiHealth RAG Chatbot",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 BobiHealth RAG Chatbot")

# Sidebar — PDF upload + index rebuild ---------------------------------------
with st.sidebar:
    st.header("📄 Document Index")
    uploaded_pdfs = st.file_uploader(
        "Upload PDF(s) to index",
        type=["pdf"],
        accept_multiple_files=True,
        help="Multiple selection allowed. They will *not* be saved permanently on Streamlit Cloud;\n         for production put your PDFs in storage and load from there.",
    )

    build_clicked = st.button("🔄 Build / Update Index", type="primary")
    clear_clicked = st.button("🗑️ Clear Conversation", type="secondary")

# -----------------------------------------------------------------------------
# 4. Build / reload FAISS + chain when needed
# -----------------------------------------------------------------------------

if build_clicked:
    if not uploaded_pdfs:
        st.error("Please upload at least one PDF before building the index.")
    else:
        with st.spinner("Building index (this might take a minute)…"):
            tmp_paths = [_write_uploaded_file(f) for f in uploaded_pdfs]
            rag_chain, vectordb = _build_index(tmp_paths)
            st.session_state["rag_chain"] = rag_chain
            st.session_state["vectordb"] = vectordb
        st.success("Index ready! Start chatting in the main panel →")

# If index already saved locally & not yet in session, load it ----------------
if "rag_chain" not in st.session_state and Path("faiss_index").exists():
    with st.spinner("Loading existing index…"):
        embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            model=aEMBED_MODEL,
            chunk_size=2048,
        )
        vectordb = FAISS.load_local("faiss_index", embeddings)
        retriever = vectordb.as_retriever(search_kwargs={"k": 4})
        # reuse helper to build chain without rebuilding embeddings
        llm = AzureChatOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            openai_api_version=API_VERSION,
            azure_deployment=CHAT_DEPLOYMENT,
            temperature=0,
            max_tokens=4096,
        )
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question",
            output_key="answer",
            return_messages=True,
        )
        prompt_template = (
            "You are a helpful assistant for BobiHealth.\n\n"
            "Use the following context to answer the user's question.\n"
            "If you don't know the answer, say you don't know—don't fabricate.\n"
            "Response should not be in more than 30‑50 words.\n\n"
            "{context}\n\n"
            "Chat History:\n{chat_history}\n\n"
            "User question: {question}\n"
            "Helpful answer:"
        )
        prompt = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=prompt_template,
        )
        rag_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True,
            output_key="answer",
        )
        st.session_state["rag_chain"] = rag_chain
        st.session_state["vectordb"] = vectordb

# Clear conversation ---------------------------------------------------------
if clear_clicked:
    for key in ("messages", "rag_chain"):
        if key in st.session_state:
            del st.session_state[key]
    st.experimental_rerun()

# -----------------------------------------------------------------------------
# 5. Chat UI
# -----------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state["messages"] = []  # list of {role: str, content: str}

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_prompt = st.chat_input("Ask something about the indexed documents…")

if user_prompt:
    if "rag_chain" not in st.session_state:
        st.warning("Please build / load the index first from the sidebar.")
    else:
        # Echo user message ----------------------------------------------------
        st.session_state["messages"].append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Query RAG chain ------------------------------------------------------
        with st.spinner("Thinking…"):
            result = st.session_state["rag_chain"].invoke({"question": user_prompt})
            answer = textwrap.fill(result["answer"], width=90)

        # Display assistant response -----------------------------------------
        with st.chat_message("assistant"):
            st.markdown(answer)
            with st.expander("Sources"):
                for doc in result["source_documents"]:
                    src = os.path.basename(doc.metadata.get("source", "unknown"))
                    page = doc.metadata.get("page", "?")
                    snippet = doc.page_content[:160].replace("\n", " ")
                    st.markdown(f"**{src}** (p.{page}) — {snippet}…")

        st.session_state["messages"].append({"role": "assistant", "content": answer})
