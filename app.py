# ─────────────────────────────────────────────────────────────────────────────
#  BobiHealth RAG Chat – Streamlit edition
# ─────────────────────────────────────────────────────────────────────────────
import os, glob, textwrap, tempfile, pickle, time
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter    import RecursiveCharacterTextSplitter
from langchain.vectorstores     import FAISS
from langchain.memory           import ConversationBufferMemory
from langchain.prompts          import PromptTemplate
from langchain.chains           import ConversationalRetrievalChain
from langchain_openai           import AzureChatOpenAI, AzureOpenAIEmbeddings

# ╭──────────────────────────────────────── SET-UP ───────────────────────────╮
st.set_page_config(page_title="BobiHealth RAG", page_icon="🩺", layout="wide")
st.title("🩺 BobiHealth document assistant")

# Read secrets (NEVER hard-code keys)
AZURE_ENDPOINT   = st.secrets["AZURE_ENDPOINT"]
AZURE_API_KEY    = st.secrets["AZURE_API_KEY"]
API_VERSION      = st.secrets.get("AZURE_API_VERSION", "2024-12-01-preview")
CHAT_DEPLOYMENT  = st.secrets.get("CHAT_DEPLOYMENT", "gpt-4o")
EMBED_MODEL      = st.secrets.get("EMBED_MODEL"    , "text-embedding-3-large")

# Session-state helpers -------------------------------------------------------
if "vectordb" not in st.session_state:    st.session_state.vectordb = None
if "chat_memory" not in st.session_state: st.session_state.chat_memory = None
if "chat_chain"  not in st.session_state: st.session_state.chat_chain  = None

# ╭──────────────────────────── Sidebar – Index builder ──────────────────────╮
st.sidebar.header("🗂️ Document index")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF(s) to index", type="pdf", accept_multiple_files=True
)

if st.sidebar.button("🔄 (Re)build index", disabled=not uploaded_files):
    with st.spinner("Embedding & indexing… this can take a minute ⏳"):
        # 1. Save PDFs to a temp dir so PyPDFLoader can open them
        with tempfile.TemporaryDirectory() as pdf_dir:
            for up in uploaded_files:
                path = os.path.join(pdf_dir, up.name)
                with open(path, "wb") as f: f.write(up.getbuffer())

            # 2. Load & split
            pdf_paths = glob.glob(os.path.join(pdf_dir, "**/*.pdf"), recursive=True)
            docs = []
            for p in pdf_paths:
                docs.extend(PyPDFLoader(p).load_and_split())

            splitter  = RecursiveCharacterTextSplitter(chunk_size=3200, chunk_overlap=800)
            splits    = splitter.split_documents(docs)

            # 3. Embed & build FAISS
            embeddings = AzureOpenAIEmbeddings(
                azure_endpoint = AZURE_ENDPOINT,
                api_key        = AZURE_API_KEY,
                model          = EMBED_MODEL,
                chunk_size     = 2048,
            )
            st.session_state.vectordb = FAISS.from_documents(splits, embeddings)
            st.session_state.chat_memory = ConversationBufferMemory(
                memory_key="chat_history", input_key="question",
                output_key="answer", return_messages=True
            )
            st.success(f"Indexed {len(splits):,} chunks from {len(pdf_paths)} PDFs.")

            # 4. Persist index to a pickle (optional – survives code reloads)
            with open(".faiss_store.pkl", "wb") as f:
                pickle.dump(st.session_state.vectordb, f)

    st.experimental_rerun()

# Load cached index (if built previously during session / from pickle) -------
if st.session_state.vectordb is None and os.path.exists(".faiss_store.pkl"):
    with open(".faiss_store.pkl", "rb") as f:
        st.session_state.vectordb = pickle.load(f)
        st.session_state.chat_memory = ConversationBufferMemory(
            memory_key="chat_history", input_key="question",
            output_key="answer", return_messages=True
        )

# Abort if no index yet -------------------------------------------------------
if st.session_state.vectordb is None:
    st.info("↖️ Upload PDFs and click **(Re)build index** to start.")
    st.stop()

# ╭────────────────────────────── Chat chain set-up ──────────────────────────╮
if st.session_state.chat_chain is None:
    llm = AzureChatOpenAI(
        azure_endpoint     = AZURE_ENDPOINT,
        api_key            = AZURE_API_KEY,
        openai_api_version = API_VERSION,
        azure_deployment   = CHAT_DEPLOYMENT,
        max_tokens         = 4096,
        temperature        = 0,
    )
    prompt_template = """You are a helpful assistant for BobiHealth.

Use the context below to answer the user question.
If the answer is not in the documents, say you don't know.
Keep responses concise (30–50 words).

{context}

Chat history:
{chat_history}

User question: {question}
Helpful answer:"""
    prompt = PromptTemplate(
        input_variables=["context","chat_history","question"],
        template=prompt_template,
    )
    st.session_state.chat_chain = ConversationalRetrievalChain.from_llm(
        llm                     = llm,
        retriever               = st.session_state.vectordb.as_retriever(search_kwargs={"k":4}),
        memory                  = st.session_state.chat_memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents = True,
        output_key              = "answer",
    )

# ╭────────────────────────────── Chat interface ─────────────────────────────╮
user_q = st.chat_input(placeholder="Ask about the uploaded documents…")
if user_q:
    with st.spinner("Thinking…"):
        result = st.session_state.chat_chain.invoke({"question": user_q})
        answer = result["answer"]

    # assistant response
    with st.chat_message("assistant"):
        st.markdown(textwrap.fill(answer, width=100))

        # show sources collapsible
        with st.expander("📄 Sources"):
            for doc in result["source_documents"]:
                src  = os.path.basename(doc.metadata.get("source", "unknown"))
                page = doc.metadata.get("page", "?")
                snippet = doc.page_content[:180].replace("\n"," ")
                st.markdown(f"* **{src}**, page {page}: {snippet}…")

# display chat history retroactively (Streamlit v1.33+ preserves messages)
