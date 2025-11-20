#debug.py
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_chroma import Chroma

# 1️⃣  Load embedding model (this will download a ~80 MB model the first time)
emb = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
print("✅ Embedding model loaded")

# 2️⃣  Connect to Ollama (make sure `ollama serve` is running)
llm = ChatOllama(model="llama3")
print("✅ LLM client created")

# 3️⃣  Connect to the existing Chroma store (replace path if different)
vector_store = Chroma(
    collection_name="esg_collection",
    embedding_function=emb,
    persist_directory="chroma_db",
)
print("✅ Connected to Chroma store – collection contains:",
      vector_store._client.get_collection("esg_collection").count(), "records")