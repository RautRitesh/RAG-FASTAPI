# run_full_rag_test.py

import os
import qdrant_client
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.cohere import Cohere
from dotenv import load_dotenv

def run_rag_test():
    """
    A complete end-to-end test that cleans, ingests, and queries the RAG system.
    """
    # --- 1. Load Environment Variables ---
    load_dotenv()
    COHERE_API_KEY = os.getenv("CO_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

    if not all([COHERE_API_KEY, QDRANT_URL, QDRANT_API_KEY]):
        print("❌ ERROR: One or more environment variables are missing.")
        return

    print("✅ Environment variables loaded.")
    
    # --- 2. Connect to Qdrant ---
    client = qdrant_client.QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    print("✅ Connected to Qdrant.")

    # --- 3. Clean and Re-Ingest Data ---
    COLLECTION_NAME = "second"
    
    # (A) Delete the old collection to ensure a fresh start
    try:
        print(f"🧹 Deleting old collection '{COLLECTION_NAME}' to ensure a clean slate...")
        client.delete_collection(collection_name=COLLECTION_NAME)
        print(f"✅ Old collection '{COLLECTION_NAME}' deleted.")
    except Exception:
        print(f"⚠️  Could not delete collection (it may not have existed, which is fine).")

    # (B) Load documents from the './data' folder
    try:
        file_path="C://Users//RiteshRaut//fast_api//rag_1//special_vishal.pdf"
        documents=SimpleDirectoryReader(
        input_files=[file_path],
        filename_as_id=True).load_data()
        print(f"✅ Loaded {len(documents)} document(s) from the 'data' folder.")
    except Exception as e:
        print(f"❌ ERROR: Could not load documents from './data'. Please create the folder and add your PDF. Error: {e}")
        return

    # (C) Ingest the documents with the correct embedding model for storage
    print("🧠 Ingesting documents with 'search_document' input type...")
    ingest_embed_model = CohereEmbedding(
        model_name="embed-english-v3.0",
        api_key=COHERE_API_KEY,
        input_type="search_document"  # CRITICAL for ingestion
    )
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=ingest_embed_model,
        show_progress=True
    )
    print("✅🎉 Ingestion complete.")
    print("-" * 50)

    # --- 4. Build Query Engine with Correct Querying Model ---
    print("🚀 Initializing query engine...")
    
    # Use the same vector_store, which is now populated
    # Initialize a NEW embed_model instance specifically for querying
    query_embed_model = CohereEmbedding(
        model_name="embed-english-v3.0",
        api_key=COHERE_API_KEY,
        input_type="search_query"  # CRITICAL for querying
    )
    
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=query_embed_model
    )
    
    llm = Cohere(model="command-r-plus", api_key=COHERE_API_KEY)
    query_engine = index.as_query_engine(llm=llm)
    
    print("✅ Query engine is ready.")
    print("-" * 50)

    # --- 5. Run a Test Query ---
    question = "what is acne?" # Change this to a relevant question for your data
    print(f"You: {question}")

    response = query_engine.query(question)
    
    print(f"🤖 Bot: {response}")
    print("-" * 50)

if __name__ == "__main__":
    run_rag_test()
