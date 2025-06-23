# ingest.py  (run locally/Colab, NOT on Streamlit Cloud)

import os, glob, pickle
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter    import RecursiveCharacterTextSplitter
from langchain_openai           import AzureOpenAIEmbeddings
from langchain.vectorstores     import FAISS

AZURE_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_API_KEY  = os.environ["AZURE_OPENAI_API_KEY"]

PDF_DIR = ""
EMBED_MODEL = "text-embedding-3-large"

# 1. Load & split
docs, splitter = [], RecursiveCharacterTextSplitter(chunk_size=3200, chunk_overlap=800)
for path in glob.glob(os.path.join(PDF_DIR, "**/*.pdf"), recursive=True):
    docs.extend(PyPDFLoader(path).load_and_split())
splits = splitter.split_documents(docs)

# 2. Embed & save
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint = AZURE_ENDPOINT,
    api_key        = AZURE_API_KEY,
    model          = EMBED_MODEL,
    chunk_size     = 2048,
)
db = FAISS.from_documents(splits, embeddings)
db.save_local("faiss_index")         # creates index.faiss & index.json
print("✅  FAISS index built & saved.")
