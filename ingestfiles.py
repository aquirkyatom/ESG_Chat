from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from uuid import uuid4
from dotenv import load_dotenv

load_dotenv()

# ==============================================================================
#                      START: CONFIGURATION
# ==============================================================================

# --- THE CRITICAL CHANGE: Define a LIST of folders to process ---
# Now you can add as many specific folders as you want.
# The script will process all of them.
DATA_FOLDERS = [
    r"data/data/frameworks",
    r"data/data/esg reports sp500",
    r"data/data/esg reports non sp500",
]
CHROMA_PATH = r"chroma_db"

print(f"--- Data Ingestion Pipeline ---")
print(f"Target data folders: {DATA_FOLDERS}")
print(f"Vector DB path:      {CHROMA_PATH}")


# --- Initiate the local, open-source embeddings model ---
print("\nLoading local embedding model...")
embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
print("Embedding model loaded.")

# --- Initiate the vector store ---
vector_store = Chroma(
    collection_name="esg_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

# ==============================================================================
#          START: MODIFIED PDF LOADING LOGIC
# ==============================================================================

# --- Loop through each specified folder and load the documents ---
print(f"\nLoading PDF documents from specified folders...")
all_raw_documents = []
for folder_path in DATA_FOLDERS:
    print(f"  -> Loading from: {folder_path}")
    loader = PyPDFDirectoryLoader(folder_path)
    # The .load() method returns a list of "Document" objects (one per page)
    documents_from_folder = loader.load()
    all_raw_documents.extend(documents_from_folder)
    print(f"     Loaded {len(documents_from_folder)} pages.")

print(f"\nTotal pages loaded from all folders: {len(all_raw_documents)}")

# ==============================================================================
#          END: MODIFIED PDF LOADING LOGIC
# ==============================================================================


# --- Splitting the documents into chunks ---
print("Splitting documents into smaller chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)
chunks = text_splitter.split_documents(all_raw_documents)
print(f"Created {len(chunks)} text chunks.")

# --- Creating unique ID's for each chunk ---
uuids = [str(uuid4()) for _ in range(len(chunks))]

# --- Adding chunks to the vector store ---
print(f"\nAdding {len(chunks)} chunks to the Chroma vector store...")
vector_store.add_documents(documents=chunks, ids=uuids)
print("--- Ingestion Complete ---")
print(f"The vector database is now saved in the '{CHROMA_PATH}' directory.")