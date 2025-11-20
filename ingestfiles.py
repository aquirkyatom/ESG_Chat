from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import SpacyTextSplitter  #spacy gives improvement over RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from uuid import uuid4
from dotenv import load_dotenv

load_dotenv()

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

#  Loop through each specified folder and load the documents --
print(f"\nLoading PDF documents from specified folders...")
all_raw_documents = []
for folder_path in DATA_FOLDERS:
    print(f"  -> Loading from: {folder_path}")
    loader = PyPDFDirectoryLoader(folder_path)
    documents_from_folder = loader.load()
    all_raw_documents.extend(documents_from_folder)
    print(f"     Loaded {len(documents_from_folder)} pages.")

print(f"\nTotal pages loaded from all folders: {len(all_raw_documents)}")


print("Splitting documents into smaller chunks...")
text_splitter = SpacyTextSplitter(
    chunk_size=1024,  # You can often use a slightly larger chunk size with Spacy because the chunks are higher quality.
    chunk_overlap=200,
    pipeline="en_core_web_lg" # This tells the splitter to use the model you just downloaded.
)
chunks = text_splitter.split_documents(all_raw_documents)
print(f"Created {len(chunks)} text chunks.")

#Creating unique ID's for each chunk
uuids = [str(uuid4()) for _ in range(len(chunks))]



print(f"\nAdding {len(chunks)} chunks to the Chroma vector store in batches...")

# Define a batch size that is safely under the ChromaDB limit of 5461
batch_size = 4096 
total_batches = (len(chunks) - 1) // batch_size + 1

for i in range(0, len(chunks), batch_size):
    # Calculate the end index for the current batch
    end_index = i + batch_size
    
    # Get the slice of chunks and their corresponding IDs for this batch
    batch_chunks = chunks[i:end_index]
    batch_uuids = uuids[i:end_index]
    
    print(f"  -> Adding batch {i//batch_size + 1}/{total_batches} ({len(batch_chunks)} chunks)...")
    
    # Add just this small batch to the vector store
    vector_store.add_documents(documents=batch_chunks, ids=batch_uuids)



print("\n--- Ingestion Complete ---")
print(f"The vector database is now saved in the '{CHROMA_PATH}' directory.")