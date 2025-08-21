# RAG with LangChain and Pinecone

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline using:
- **LangChain** for document loading, chunking, embeddings, and orchestration
- **OpenAI** for embeddings and LLM responses
- **Pinecone** as the vector database for storage and similarity search

---

## 🔄 Flow Overview

### 1. Document Ingestion
- **Loader**: `PyPDFDirectoryLoader` reads raw PDF files into LangChain `Document` objects.  
- **Chunking**: `RecursiveCharacterTextSplitter` splits documents into overlapping text chunks (e.g., 1000 tokens with 200 overlap).  

### 2. Embedding and Storage
- **Embeddings**: `OpenAIEmbeddings` converts each chunk into a dense vector (1536 or 3072 dimensions depending on model).  
- **Vector Store**: `PineconeVectorStore.from_documents` uploads embeddings + metadata into a Pinecone index.  

### 3. Query Workflow
- **User Query**: User asks a natural language question.  
- **Query Embedding**: Same `OpenAIEmbeddings` model embeds the query.  
- **Vector Search**: Pinecone returns the top-k most relevant chunks.  
- **Prompt Assembly**: LangChain’s `RetrievalQA` combines retrieved chunks with the query.  
- **LLM Answer**: `ChatOpenAI` generates a grounded response using both the query and retrieved context.  

---

## 📊 Flow Diagram

<img width="1589" height="1314" alt="output (9)" src="https://github.com/user-attachments/assets/cca513ca-2817-475b-9e47-6a7c636b59f4" />


---

## 🚀 Usage

### 1. Install Dependencies
```bash
python -m pip install -r requirements.txt
