# --- IMPORTS FOR LOCAL MODELS ---
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.chat_models import ChatOllama
# --------------------------------

from langchain_chroma import Chroma
import gradio as gr

# Configuration
CHROMA_PATH = r"chroma_db"

# ==============================================================================
#                      START: SWAP TO LOCAL MODELS
# ==============================================================================

# Load the LOCAL embedding model ---

print("Loading local embedding model...")
embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
print("Embedding model loaded.")

# --- 2. Initiate the LOCAL, open-source LLM via Ollama ---
print("Initiating local LLM (e.g., Llama 3)...")
llm = ChatOllama(model="llama3")
print("LLM initiated.")




# --- Connect to the existing chromadb you created with the ingest script ---
print("Connecting to the Chroma vector store...")
vector_store = Chroma(
    collection_name="esg_collection", # Use the same name as in your ingest script
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH, 
)
print("Connected to vector store.")

# This will find the most relevant chunks from your database for a given question.
num_results = 5
retriever = vector_store.as_retriever(search_kwargs={'k': num_results})

# -core RAG function 
def stream_response(message, history):
    print(f"\nUser query: {message}")

    # 1. Retrieve relevant chunks from your knowledge base
    docs = retriever.invoke(message)
    
    # 2. Format the retrieved knowledge
    knowledge = ""
    for i, doc in enumerate(docs):
        knowledge += f"--- Document {i+1} ---\n" + doc.page_content + "\n\n"
    
    print(f"Retrieved {len(docs)} relevant document chunks.")

    # 3. Create the final prompt for the local LLM
    if message:
        partial_message = ""

        # This is a good, standard prompt for a RAG system.
        rag_prompt = f"""
        You are an expert ESG assistant. Your task is to answer the user's question based *only* on the provided "Knowledge Base" section.
        Do not use any of your own internal knowledge. If the answer is not found in the knowledge base, state that clearly.
        Do not mention the existence of the knowledge base to the user.

        --- Knowledge Base ---
        {knowledge}
        --- End of Knowledge Base ---

        Conversation history: {history}

        User's Question: {message}

        Answer:
        """

        # 4. Stream the response from the local LLM
        for response_chunk in llm.stream(rag_prompt):
            # `response_chunk` is a string with the next piece of the answer
            partial_message += response_chunk
            yield partial_message

# --- Initiate and launch the Gradio user interface ---
print("\nLaunching Gradio Chat Interface...")
chatbot = gr.ChatInterface(
    stream_response, 
    chatbot=gr.Chatbot(height=500),
    textbox=gr.Textbox(placeholder="Ask a question about an ESG policy...", container=False, scale=7),
    title="ESG Policy Chatbot",
    description="This chatbot answers questions about ESG policies using a local knowledge base.",
    theme="soft",
    examples=[
        ["What are the key reporting metrics for GHG emissions?"],
        ["Summarize the SASB standard for the software industry."],
        ["What is a good policy for board diversity?"]
    ]
)

chatbot.launch()