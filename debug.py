#debug.py
#Ollama server is running (ollama serve).	
#Model llama3 (or another you prefer) is pulled (ollama pull llama3).	
#Python env has all packages from the requirements.txt we discussed.	
#Chroma DB exists at chroma_db and contains a collection named esg_collection.	
#GPU (optional) is available if you installed the GPU‑enabled Ollama; otherwise stay on CPU.	
#Try a RAG query – e.g., “What are the key reporting metrics for GHG emissions?” – you should see citations like [Document 1].	
#Try a generic query – e.g., “Tell me a joke about climate change.” – you should get a normal Llama response without any citation.
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import SpacyTextSplitter

# ==============================================================================
#                      CONFIGURATION
# ==============================================================================
# Use a specific, small folder for debugging to make it fast.
# Choose a folder with just one or two small PDFs.
DEBUG_DATA_FOLDER = r"data/frameworks" 

# How many items to show at each step
N_SAMPLES_TO_SHOW = 2

# ==============================================================================
#          STEP 1: DEBUG THE DOCUMENT LOADER (PyPDF)
# ==============================================================================
print("="*60)
print("--- DEBUGGING STEP 1: PyPDFDirectoryLoader ---")
print("="*60)

if not os.path.exists(DEBUG_DATA_FOLDER):
    print(f"FATAL ERROR: The debug directory was not found at '{DEBUG_DATA_FOLDER}'")
else:
    # --- Load the raw documents ---
    print(f"Loading all PDF documents from: {DEBUG_DATA_FOLDER}...")
    loader = PyPDFDirectoryLoader(DEBUG_DATA_FOLDER)
    raw_documents = loader.load()
    print(f"Loaded a total of {len(raw_documents)} pages.")

    # --- Inspect the output ---
    print(f"\n--- Here are the first {N_SAMPLES_TO_SHOW} loaded 'Document' objects: ---")
    for i, doc in enumerate(raw_documents[:N_SAMPLES_TO_SHOW]):
        print(f"\n--- Document {i+1} ---")
        
        # 1. Print the text content (first 400 characters)
        print(">>> Page Content (first 400 chars):")
        print(doc.page_content[:400].replace('\n', ' '))
        
        # 2. Print the metadata (source file and page number)
        print("\n>>> Metadata:")
        print(doc.metadata)

# ==============================================================================
#          STEP 2: DEBUG THE TEXT SPLITTER (SpaCy)
# ==============================================================================
    print("\n\n" + "="*60)
    print("--- DEBUGGING STEP 2: SpacyTextSplitter ---")
    print("="*60)

    # --- Initialize the splitter ---
    text_splitter = SpacyTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        pipeline="en_core_web_lg"
    )

    # --- Split the documents into chunks ---
    print("Splitting the loaded documents into chunks...")
    chunks = text_splitter.split_documents(raw_documents)
    print(f"Created a total of {len(chunks)} text chunks.")

    # --- Inspect the output ---
    print(f"\n--- Here are the first {N_SAMPLES_TO_SHOW} chunks created from the first document: ---")
    # To make this clearer, we'll just split the first document
    first_doc_chunks = text_splitter.split_documents([raw_documents[0]])
    
    for i, chunk in enumerate(first_doc_chunks[:N_SAMPLES_TO_SHOW]):
        print(f"\n--- Chunk {i+1} ---")
        
        # 1. Print the text content of the chunk
        print(">>> Chunk Content:")
        print(chunk.page_content.replace('\n', ' '))
        
        # 2. Print the length of the chunk
        print(f"\n>>> Length of this chunk: {len(chunk.page_content)} characters")

        # 3. Print the metadata (it should be the same as the source document)
        print("\n>>> Metadata:")
        print(chunk.metadata)

    print("\n--- Debugging Complete ---")