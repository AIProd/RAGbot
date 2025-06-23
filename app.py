# streamlit_app.py
import os
import glob
import textwrap
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.prompts import PromptTemplate

# ╔═════════════════════════════════════════════════════════════════════╗
# ║ Streamlit UI                                                        ║
# ╚═════════════════════════════════════════════════════════════════════╝
st.set_page_config(page_title="BobiHealth RAGbot", page_icon="🧵")
st.title("BobiHealth Chatbot")
st.markdown("Ask questions from your medical PDFs.")

# ╔═════════════════════════════════════════════════════════════════════╗
# ║ AZURE OPENAI CONFIG                                                 ║
# ╚═════════════════════════════════════════════════════════════════════╝
AZURE_ENDPOINT   = st.secrets["AZURE_OPENAI_ENDPOINT"]
AZURE_API_KEY    = st.secrets["AZURE_OPENAI_API_KEY"]
API_VERSION      = "2024-12-01-preview"
CHAT_DEPLOYMENT  = "gpt-4o"
EMBED_MODEL      = "text-embedding-3-large"

# ╔═════════════════════════════════════════════════════════════════════╗
# ║ Load FAISS Index                                                    ║
# ╚═════════════════════════════════════════════════════════════════════╝
db = FAISS.load_local("faiss_index", embeddings=AzureOpenAIEmbeddings(
    azure_endpoint = AZURE_ENDPOINT,
    api_key        = AZURE_API_KEY,
    model          = EMBED_MODEL
))
retriever = db.as_retriever(search_kwargs={"k": 4})

# ╔═════════════════════════════════════════════════════════════════════╗
# ║ LLM & Memory Setup                                                  ║
# ╚═════════════════════════════════════════════════════════════════════╝
llm = AzureChatOpenAI(
    azure_endpoint     = AZURE_ENDPOINT,
    api_key            = AZURE_API_KEY,
    openai_api_version = API_VERSION,
    azure_deployment   = CHAT_DEPLOYMENT,
    max_tokens         = 4096,
    temperature        = 0
)

memory = ConversationBufferMemory(
    memory_key      = "chat_history",
    input_key       = "question",
    output_key      = "answer",
    return_messages = True
)

prompt_template = """You are a helpful assistant for BobiHealth.

Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say so — don't make it up.

{context}

Chat History:
{chat_history}

Question: {question}
Helpful Answer:"""

prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=prompt_template
)

rag_bot = ConversationalRetrievalChain.from_llm(
    llm                     = llm,
    retriever               = retriever,
    memory                  = memory,
    combine_docs_chain_kwargs={"prompt": prompt},
    return_source_documents = True,
    output_key              = "answer"
)

# ╔═════════════════════════════════════════════════════════════════════╗
# ║ Chat Interface                                                      ║
# ╚═════════════════════════════════════════════════════════════════════╝
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_q = st.chat_input("Ask something from the documents...")

if user_q:
    with st.spinner("Thinking..."):
        result = rag_bot.invoke({"question": user_q})
        answer = result["answer"]
        sources = result["source_documents"]

        st.session_state.chat_history.append((user_q, answer, sources))

# Display history
for q, a, source_docs in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        st.markdown(a)
        if source_docs:
            st.markdown("**Sources:**")
            for doc in source_docs:
                src = os.path.basename(doc.metadata.get("source", "unknown"))
                page = doc.metadata.get("page", "?")
                snippet = doc.page_content[:120].replace("\n", " ")
                st.markdown(f"- `{src}` (p.{page}): {snippet}...")
