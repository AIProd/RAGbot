# ──────────────────────────────────────────────────────────────────────────────
# app.py  ·  BobiHealth RAG Chatbot  ·  Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────
#  Requires: streamlit, langchain, langchain-community, langchain-openai,
#            faiss-cpu, pypdf, tiktoken, cryptography>=42
#
#  Put your Azure OpenAI keys in .streamlit/secrets.toml:
#    AZURE_ENDPOINT = "https://<your-endpoint>.cognitiveservices.azure.com/"
#    AZURE_API_KEY  = "sk-..."
#    API_VERSION    = "2024-12-01-preview"
# ──────────────────────────────────────────────────────────────────────────────
import os
import base64
import tempfile
import textwrap
from pathlib import Path
from typing import List

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Secrets / env config ─ add via Streamlit → Settings → Secrets
# ──────────────────────────────────────────────────────────────────────────────
AZURE_ENDPOINT = st.secrets.get("AZURE_ENDPOINT", os.getenv("AZURE_ENDPOINT"))
AZURE_API_KEY  = st.secrets.get("AZURE_API_KEY",  os.getenv("AZURE_API_KEY"))
API_VERSION    = st.secrets.get("API_VERSION",
                                os.getenv("API_VERSION", "2024-12-01-preview"))

CHAT_DEPLOYMENT = "gpt-4o"                     # chat completion deployment name
EMBED_MODEL     = "text-embedding-3-large"     # embedding model / deployment

# ──────────────────────────────────────────────────────────────────────────────
# 2.  Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _write_uploaded_file(uploaded_file) -> Path:
    """Persist Streamlit UploadedFile to a temp path and return it."""
    suffix = Path(uploaded_file.name).suffix
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(uploaded_file.read())
    return Path(tmp_path)


def _build_index(pdf_paths: List[Path]):
    """Embed the provided PDFs and build FAISS index → RAG chain."""
    # 2·1  Load & split
    docs = []
    for p in pdf_paths:
        docs.extend(PyPDFLoader(str(p)).load_and_split())
    splitter = RecursiveCharacterTextSplitter(chunk_size=3200, chunk_overlap=800)
    splits = splitter.split_documents(docs)

    # 2·2  Embed & store
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        model=EMBED_MODEL,
        chunk_size=2048,
    )
    vectordb = FAISS.from_documents(splits, embeddings)
    vectordb.save_local("faiss_index")  # persist between sessions

    # 2·3  Build RAG chain
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
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
        "Response should not be in more than 30-50 words.\n\n"
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

# ──────────────────────────────────────────────────────────────────────────────
# 3.  Streamlit layout
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="BobiHealth RAG Chatbot",
                   page_icon="bobihealth_logo.jfif", layout="centered")
with open("icon.png", "rb") as image_file:
    encoded = base64.b64encode(image_file.read()).decode()

# HTML layout
st.markdown(
    f"""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
        <img src="data:image/png;base64,{encoded}" style="height:48px;">
        <h1 style="margin: 0; font-size: 2.2rem;">RAG Chatbot</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar – PDF upload / index controls
with st.sidebar:
    st.header("📄 Document Index")
    api_key_input = st.text_input("🔑 OpenAI / Azure API key", type="password",
                                  placeholder="sk-…  or  <azure-key>")
    if api_key_input:
        AZURE_API_KEY = api_key_input.strip()
    uploaded_pdfs = st.file_uploader(
        "Upload PDF(s) to index",
        type=["pdf"],
        accept_multiple_files=True,
        help="Files stay in-memory for this session. For production, "
             "load from persistent storage instead.",
    )
    build_clicked = st.button("🔄 Build / Update Index", type="primary")
    clear_clicked = st.button("🗑️ Clear Conversation", type="secondary")

# 4.  Build / load index ------------------------------------------------------
if build_clicked:
    if not uploaded_pdfs:
        st.error("Please upload at least one PDF first.")
    else:
        with st.spinner("Building index (this may take a minute)…"):
            tmp_paths = [_write_uploaded_file(f) for f in uploaded_pdfs]
            chain, db = _build_index(tmp_paths)
            st.session_state["rag_chain"] = chain
            st.session_state["vectordb"]  = db
        st.success("Index built! Chat away →")

# Auto-load existing index on cold start
if "rag_chain" not in st.session_state and Path("faiss_index").exists():
    with st.spinner("Loading existing index…"):
        emb = AzureOpenAIEmbeddings(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            model=EMBED_MODEL,
            chunk_size=2048,
        )
        db = FAISS.load_local("faiss_index", emb, allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_kwargs={"k": 4})
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
            "Response should not be in more than 30-50 words.\n\n"
            "{context}\n\n"
            "Chat History:\n{chat_history}\n\n"
            "User question: {question}\n"
            "Helpful answer:"
        )
        prompt = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=prompt_template,
        )
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True,
            output_key="answer",
        )
        st.session_state["rag_chain"] = chain
        st.session_state["vectordb"]  = db

# Clear conversation
if clear_clicked:
    for k in ("messages", "rag_chain"):
        st.session_state.pop(k, None)
    st.experimental_rerun()

# ──────────────────────────────────────────────────────────────────────────────
# 5.  Chat UI
# ──────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # [{role, content}, …]

for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask something about the indexed documents…")
if prompt:
    if "rag_chain" not in st.session_state:
        st.warning("Please build / load an index first from the sidebar.")
    else:
        # Echo user prompt
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Query chain
        with st.spinner("Thinking…"):
            result  = st.session_state["rag_chain"].invoke({"question": prompt})
            answer  = textwrap.fill(result["answer"], width=90)

        # Show assistant reply
        with st.chat_message("assistant"):
            st.markdown(answer)
            if result["source_documents"]:
                with st.expander("Sources"):
                    for doc in result["source_documents"]:
                        src  = os.path.basename(doc.metadata.get("source", "unknown"))
                        page = doc.metadata.get("page", "?")
                        snip = doc.page_content[:160].replace("\n", " ")
                        st.markdown(f"**{src}** (p.{page}) — {snip}…")

        st.session_state["messages"].append({"role": "assistant", "content": answer})
