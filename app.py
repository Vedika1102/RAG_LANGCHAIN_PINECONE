# main.py — LangChain + OpenAI + Pinecone (modern APIs)

import os
from dotenv import load_dotenv

# Load environment variables from .env (KEY=value, no quotes)
# .env must contain:
# OPENAI_API_KEY=sk-...
# PINECONE_API_KEY=pc-...
load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "Missing OPENAI_API_KEY"
assert os.getenv("PINECONE_API_KEY"), "Missing PINECONE_API_KEY"

# --- Imports (modern) ---
from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA

# --- Config ---
PDF_DIR = "documents/"        # change this
INDEX_NAME = "langchainvector"       # change or keep
EMBED_MODEL = "text-embedding-3-small"   # 1536 dims; use -3-large for 3072
REGION = "us-east-1"                 # Pinecone serverless region
CLOUD = "aws"

# Map embedding model → dimension
EMBED_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}

# --- Embeddings ---
embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

# --- Ensure Pinecone index exists ---
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
existing = {i["name"] for i in pc.list_indexes().get("indexes", [])}
if INDEX_NAME not in existing:
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIMS[EMBED_MODEL],
        metric="cosine",
        spec={"serverless": {"cloud": CLOUD, "region": REGION}},
    )

# --- Load and split documents ---
loader = PyPDFDirectoryLoader(PDF_DIR)
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(docs)

# --- Create / connect vector store and upsert documents ---
# If the index already has data and you only want to connect, use:
# vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
vectorstore = PineconeVectorStore.from_documents(
    documents=splits,
    embedding=embeddings,
    index_name=INDEX_NAME,
)

# --- Retrieval QA example ---
llm = ChatOpenAI(model="gpt-4o-mini")
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

try:
    question = "What does the document say about modeling?"
    answer = qa.run(question)
    print(f"Q: {question}\nA: {answer}")
except Exception as e:
    # Common cause: insufficient_quota from OpenAI billing
    print("Error during retrieval/LLM call:", repr(e))
    print("If this is an OpenAI quota error, enable billing or use a key with credits.")
