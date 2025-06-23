# app.py  (runs on Streamlit Cloud)

import os, textwrap, streamlit as st
from langchain.vectorstores import FAISS
from langchain_openai       import AzureChatOpenAI
from langchain.memory       import ConversationBufferMemory
from langchain.chains       import ConversationalRetrievalChain
from langchain.prompts      import PromptTemplate

# --- env vars set via Streamlit Cloud Secrets --------------------------
AZURE_ENDPOINT   = st.secrets["AZURE_OPENAI_ENDPOINT"]
AZURE_API_KEY    = st.secrets["AZURE_OPENAI_API_KEY"]
API_VERSION      = "2024-12-01-preview"
CHAT_DEPLOYMENT  = "gpt-4o"

# --- load FAISS index (no embedding calls) -----------------------------
db = FAISS.load_local("faiss_index", embeddings=None)
retriever = db.as_retriever(search_kwargs={"k": 4})

# --- LLM & memory ------------------------------------------------------
llm = AzureChatOpenAI(
    azure_endpoint     = AZURE_ENDPOINT,
    api_key            = AZURE_API_KEY,
    openai_api_version = API_VERSION,
    azure_deployment   = CHAT_DEPLOYMENT,
    max_tokens         = 4096,
    temperature        = 0,
)

memory = ConversationBufferMemory(
    memory_key      ="chat_history",
    input_key       ="question",
    output_key      ="answer",
    return_messages = True,
)

prompt = PromptTemplate(
    template = """You are a helpful assistant for BobiHealth.

{context}

Chat History:
{chat_history}

User question: {question}
Helpful answer:""",
    input_variables=["context","chat_history","question"],
)

chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt},
    output_key="answer",
    return_source_documents=True,
)

# --- Streamlit UI ------------------------------------------------------
st.set_page_config(page_title="BobiHealth RAG Chatbot")
st.title("💊 BobiHealth Assistant")

if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

for role, msg in st.session_state.chat_log:
    st.chat_message(role).write(msg)

user_msg = st.chat_input("Ask me something about the documents…")
if user_msg:
    st.chat_message("user").write(user_msg)

    result = chain.invoke({"question": user_msg})
    answer = result["answer"]

    # store history
    st.session_state.chat_log.append(("user", user_msg))
    st.session_state.chat_log.append(("assistant", answer))

    bot = st.chat_message("assistant")
    bot.write(textwrap.fill(answer, 90))

    with bot.expander("show sources"):
        for d in result["source_documents"]:
            src  = os.path.basename(d.metadata.get("source","?"))
            page = d.metadata.get("page","?")
            snippet = d.page_content[:120].replace("\n"," ")
            st.markdown(f"**{src} p.{page}** — {snippet}…")
