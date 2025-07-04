import os
import qdrant_client
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.cohere import Cohere
from dotenv import load_dotenv

# --- Pydantic Model ---
class QueryRequest(BaseModel):
    question: str

# --- Load Environment Variables & Debug ---
print("--> [1/12] Loading environment variables...")
load_dotenv()
COHERE_API_KEY = os.getenv("CO_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Check if variables are loaded
if not all([COHERE_API_KEY, QDRANT_URL, QDRANT_API_KEY]):
    print("--> FATAL ERROR: One or more environment variables are missing!")
    print(f"--> COHERE_API_KEY loaded: {'Yes' if COHERE_API_KEY else 'No'}")
    print(f"--> QDRANT_URL loaded: {'Yes' if QDRANT_URL else 'No'}")
    print(f"--> QDRANT_API_KEY loaded: {'Yes' if QDRANT_API_KEY else 'No'}")
else:
    print("--> [2/12] All environment variables loaded successfully.")

# --- Initialize FastAPI App ---
print("--> [3/12] Initializing FastAPI app...")
app = FastAPI(
    title="RAG Chatbot API",
    description="A simple API to chat with a RAG model powered by Cohere and Qdrant."
)
print("--> [4/12] FastAPI app initialized.")

# --- Add CORS Middleware ---
print("--> [5/12] Adding CORS middleware...")
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
print("--> [6/12] CORS middleware added.")

# --- One-time Setup of RAG Components (with debugging) ---
try:
    print("--> [7/12] Initializing Cohere embedding model...")
    embed_model = CohereEmbedding(
        model_name="embed-english-v3.0",
        api_key=COHERE_API_KEY
    )
    print("--> [8/12] Initializing Qdrant client...")
    client = qdrant_client.QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )
    print("--> [9/12] Initializing Qdrant vector store...")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="second",
        embed_model=embed_model
    )
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )
    print("--> [10/12] Loading index from vector store...")
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
        storage_context=storage_context
    )
    print("--> [11/12] Initializing Cohere LLM and query engine...")
    llm = Cohere(
        model="command-r-plus",
        api_key=COHERE_API_KEY
    )
    query_engine = index.as_query_engine(llm=llm)
    print("--> [12/12] SETUP COMPLETE! RAG components are ready.")

except Exception as e:
    print(f"--> FATAL ERROR during setup: {e}")


# --- API Endpoints ---
@app.get("/", summary="Root endpoint to check if the API is running")
def read_root():
    return {"message": "Welcome to the RAG Chatbot API! Setup is complete."}

@app.post("/query", summary="Ask a question to the RAG model")
async def handle_query(request: QueryRequest):
    response = query_engine.query(request.question)
    raw_text = str(response)
    formatted_text = raw_text.replace('\n', '<br>')
    return {"answer": formatted_text}