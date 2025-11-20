from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from uuid import uuid4
from dotenv import load_dotenv

load_dotenv()



print("\n--- Starting Data Ingestion Script ---, from folders data/esg reports sp500, data/esg reports non sp500 and data/frameworks so rename your folders accordingly if needed.")
#  specific folders 
DATA_FOLDERS = [
    r"data/frameworks",
    r"data/esg reports sp500",
    r"data/esg reports non sp500",
]
CHROMA_PATH = r"chroma_db"

print(f"--- Data Ingestion Pipeline ---")
print(f"Target data folders: {DATA_FOLDERS}")
print(f"Vector DB path:      {CHROMA_PATH}")


# Initiate  model 
print("\nLoading local embedding model...")
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Embedding model loaded.")

# Initiate  vector store 
vector_store = Chroma(
    collection_name="esg_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

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