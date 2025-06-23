# app.py

import os
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# -- 1. CONFIGURE KEYS --
AZURE_ENDPOINT     = st.secrets["AZURE_OPENAI_ENDPOINT"]
AZURE_API_KEY      = st.secrets["AZURE_OPENAI_API_KEY"]
API_VERSION        = "2024-12-01-preview"
CHAT_DEPLOYMENT    = "gpt-4o"
EMBED_DEPLOYMENT   = "text-embedding-3-large"
VECTORDB_PATH      = "faiss_index"   # Pre-built FAISS index folder

# -- 2. LOAD FAISS VECTOR DB --
vectordb = FAISS.load_local(VECTORDB_PATH, AzureOpenAIEmbeddings(
    azure_endpoint = AZURE_ENDPOINT,
    api_key        = AZURE_API_KEY,
    model          = EMBED_DEPLOYMENT,
    chunk_size     = 2048
))
retriever = vectordb.as_retriever(search_kwargs={"k": 4})

# -- 3. SETUP LLM + MEMORY --
llm = AzureChatOpenAI(
    azure_endpoint     = AZURE_ENDPOINT,
    api_key            = AZURE_API_KEY,
    openai_api_version = API_VERSION,
    azure_deployment   = CHAT_DEPLOYMENT,
    max_tokens         = 4096,
    temperature        = 0
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question",
    output_key="answer",
    return_messages=True
)

prompt_template = """You are a helpful assistant for BobiHealth.

Use the following context to answer the user's question.
If you don't know the answer, say so honestly.

{context}

Chat History:
{chat_history}

User question: {question}
Helpful answer:"""

prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=prompt_template
)

rag_bot = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt},
    return_source_documents=True,
    output_key="answer"
)

# -- 4. STREAMLIT UI --
st.title("🤖 BobiHealth Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask something from your medical documents:")

if user_input:
    result = rag_bot.invoke({"question": user_input})
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", result["answer"]))

    for speaker, msg in st.session_state.chat_history:
        st.markdown(f"**{speaker}:** {msg}")
