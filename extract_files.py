import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

# ==============================================================================
#                      CONFIGURATION
# ==============================================================================
# These paths MUST match the ones used in your `ingest_files.py` script.

CHROMA_PATH = r"chroma_db"
COLLECTION_NAME = "esg_collection"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OUTPUT_CSV_PATH = "esg_knowledge_base.csv"

# ==============================================================================
#          STEP 1: CONNECT TO THE EXISTING VECTOR DATABASE
# ==============================================================================
print("--- Data Extraction Script ---")

# --- Check if the database directory exists ---
if not os.path.exists(CHROMA_PATH):
    print(f"FATAL ERROR: The ChromaDB directory was not found at '{CHROMA_PATH}'")
    print("Please make sure you have run your `ingest_files.py` script successfully first.")
else:
    print(f"Connecting to existing ChromaDB at: {CHROMA_PATH}")
    
    # --- Load the local embedding model (must match the one used for ingestion) ---
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # --- Connect to the persisted vector store ---
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings_model,
        persist_directory=CHROMA_PATH,
    )
    print("Successfully connected to the vector store.")

# ==============================================================================
#          STEP 2: RETRIEVE ALL DATA FROM THE COLLECTION
# ==============================================================================
    # The .get() method is the way to retrieve all items from a Chroma collection.
    # We include "metadatas" and "documents" to get the text and source info.
    print("\nRetrieving all stored document chunks and metadata...")
    results = vector_store.get(include=["metadatas", "documents"])
    
    retrieved_ids = results['ids']
    retrieved_documents = results['documents']
    retrieved_metadatas = results['metadatas']
    
    print(f"Successfully retrieved {len(retrieved_ids)} chunks from the database.")

# ==============================================================================
#          STEP 3: CONVERT THE DATA INTO A PANDAS DATAFRAME
# ==============================================================================
    print("\nStructuring the data into a Pandas DataFrame...")
    
    # The metadata from PyPDFDirectoryLoader contains the source file and page number.
    # We will extract these into their own columns.
    source_files = [meta.get('source', 'Unknown') for meta in retrieved_metadatas]
    page_numbers = [meta.get('page', -1) for meta in retrieved_metadatas]
    
    # Create the DataFrame
    df = pd.DataFrame({
        'chunk_id': retrieved_ids,
        'text_chunk': retrieved_documents,
        'source_file': source_files,
        'page_number': page_numbers
    })
    
    print("DataFrame created successfully.")
    print("\n--- DataFrame Info ---")
    df.info()
    
    print("\n--- First 5 Rows of the DataFrame ---")
    print(df.head().to_string())

# ==============================================================================
#          STEP 4: SAVE THE DATAFRAME TO A CSV FILE
# ==============================================================================
    print(f"\nSaving the DataFrame to a CSV file at: {OUTPUT_CSV_PATH}")
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print("--- Extraction Complete ---")
    print("You can now load this CSV file into a new notebook for your machine learning tasks.")