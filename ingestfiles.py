from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import SpacyTextSplitter
from langchain_chroma import Chroma
from uuid import uuid4
from dotenv import load_dotenv
import re  # <--- NEW: For Regex pattern matching

load_dotenv()

# ==========================================
# 1. DEFINE CLEANING FUNCTION
# ==========================================
def clean_document_content(text):
    """
    Filters out Table of Contents lines, Page numbers, and short noise.
    """
    if not text:
        return ""
    
    lines = text.split('\n')
    cleaned_lines = []
    
    # Regex to find lines ending in dots and a number (e.g. "Strategy ...... 12")
    toc_pattern = re.compile(r'\.{3,}\s*\d+$')
    
    for line in lines:
        line = line.strip()
        
        # Rule 1: Skip empty lines
        if not line:
            continue
            
        # Rule 2: Skip standalone page numbers (e.g. "45")
        if line.isdigit():
            continue
            
        # Rule 3: Skip Table of Contents lines (dots followed by number)
        if toc_pattern.search(line):
            continue
            
        # Rule 4: Skip explicit header keywords
        lower_line = line.lower()
        if "table of contents" in lower_line or "index of tables" in lower_line:
            continue
            
        # Rule 5: Skip very short lines (often artifacts like "|")
        # But keep them if they look like bullet points ("- A")
        if len(line) < 4 and not line.startswith(('-', '•', '1.')):
            continue
            
        cleaned_lines.append(line)
        
    return "\n".join(cleaned_lines)

# ==========================================
# 2. MAIN PIPELINE
# ==========================================

print("\n--- Starting Data Ingestion Script ---")
DATA_FOLDERS = [
    r"data/frameworks",
    r"data/esg reports sp500",
    r"data/esg reports non sp500",
]
CHROMA_PATH = r"chroma_db"

print(f"--- Data Ingestion Pipeline ---")
print(f"Target data folders: {DATA_FOLDERS}")
print(f"Vector DB path:      {CHROMA_PATH}")

# Initiate model 
print("\nLoading local embedding model...")
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Embedding model loaded.")

# Initiate vector store
vector_store = Chroma(
    collection_name="esg_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

# Loop through each specified folder and load the documents
print(f"\nLoading PDF documents from specified folders...")
all_raw_documents = []
for folder_path in DATA_FOLDERS:
    print(f"  -> Loading from: {folder_path}")
    loader = PyPDFDirectoryLoader(folder_path)
    documents_from_folder = loader.load()
    all_raw_documents.extend(documents_from_folder)
    print(f"     Loaded {len(documents_from_folder)} pages.")

print(f"\nTotal pages loaded: {len(all_raw_documents)}")

# ==========================================
# 3. APPLY CLEANING (NEW STEP)
# ==========================================
print("\nCleaning documents (removing TOCs, page numbers, artifacts)...")
cleaned_count = 0
for doc in all_raw_documents:
    original_len = len(doc.page_content)
    # Apply the function directly to the page_content attribute
    doc.page_content = clean_document_content(doc.page_content)
    
    if len(doc.page_content) < original_len:
        cleaned_count += 1

print(f"Cleaned content in {cleaned_count} pages.")


# ==========================================
# 4. SPLITTING & EMBEDDING
# ==========================================
print("Splitting documents into smaller chunks...")
text_splitter = SpacyTextSplitter(
    chunk_size=1024,
    chunk_overlap=200,
    pipeline="en_core_web_lg"
)

# Note: SpacySplitter works much better now that "Page 15" isn't interrupting sentences
chunks = text_splitter.split_documents(all_raw_documents)
print(f"Created {len(chunks)} text chunks.")

#Creating unique ID's for each chunk
uuids = [str(uuid4()) for _ in range(len(chunks))]

print(f"\nAdding {len(chunks)} chunks to the Chroma vector store in batches...")

# Define a batch size that is safely under the ChromaDB limit
batch_size = 4096 
total_batches = (len(chunks) - 1) // batch_size + 1

for i in range(0, len(chunks), batch_size):
    end_index = i + batch_size
    batch_chunks = chunks[i:end_index]
    batch_uuids = uuids[i:end_index]
    
    print(f"  -> Adding batch {i//batch_size + 1}/{total_batches} ({len(batch_chunks)} chunks)...")
    vector_store.add_documents(documents=batch_chunks, ids=batch_uuids)

print("\n--- Ingestion Complete ---")
print(f"The vector database is now saved in the '{CHROMA_PATH}' directory.")