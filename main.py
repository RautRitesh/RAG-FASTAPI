# main.py

import os
import qdrant_client
from fastapi import FastAPI
from pydantic import BaseModel
# --- IMPORT THE CORS MIDDLEWARE ---
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.cohere import Cohere
from dotenv import load_dotenv

# --- 1. Pydantic Model for Request Body ---
class QueryRequest(BaseModel):
    question: str

# --- 2. Load Environment Variables ---
load_dotenv()
COHERE_API_KEY = os.getenv("CO_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# --- 3. Initialize FastAPI App ---
app = FastAPI(
    title="RAG Chatbot API",
    description="A simple API to chat with a RAG model powered by Cohere and Qdrant."
)

# --- 4. ADD THE CORS MIDDLEWARE ---
# This section allows your frontend (running on any domain) to communicate with your backend.
origins = [
    "*",  # Allows all origins
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


# --- 5. One-time Setup of RAG Components ---
# This code runs only once when the FastAPI server starts up.

# Initialize Cohere embedding model
embed_model = CohereEmbedding(
    model_name="embed-english-v3.0",
    api_key=COHERE_API_KEY
)

# Initialize Qdrant client
client = qdrant_client.QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# Initialize the Qdrant Vector Store
vector_store = QdrantVectorStore(
    client=client,
    collection_name="second",
    embed_model=embed_model
)

# Initialize the Storage Context
storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)

# Load the index from the vector store
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=embed_model,
    storage_context=storage_context
)

# Initialize the Cohere LLM
llm = Cohere(
    model="command-r-plus",
    api_key=COHERE_API_KEY
)

# Create the query engine
query_engine = index.as_query_engine(llm=llm)


# --- 6. API Endpoints ---

@app.get("/", summary="Root endpoint to check if the API is running")
def read_root():
    """A simple endpoint to confirm that the server is up and running."""
    return {"message": "Welcome to the RAG Chatbot API!"}


@app.post("/query", summary="Ask a question to the RAG model")
async def handle_query(request: QueryRequest):
    """
    This endpoint receives a question, passes it to the RAG query engine,
    and returns the model's answer with preserved formatting.
    """
    response = query_engine.query(request.question)
    
    # Access the raw text from the response object
    raw_text = str(response)
    
    # Replace newline characters with HTML line breaks for better frontend rendering.
    # The frontend's use of `innerHTML` will render these breaks correctly.
    formatted_text = raw_text.replace('\n', '<br>')
    
    return {"answer": formatted_text}

