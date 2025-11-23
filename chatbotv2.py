# ==================== ESG‑Chatbot script ====================

# 0️⃣  Imports -------------------------------------------------
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma   
import gradio as gr
import numpy as np

# 1️⃣  Configuration -------------------------------------------
CHROMA_PATH       = r"chroma_db"        # folder that already contains your persisted DB
COLLECTION_NAME   = "esg_collection"    # must match the name used in the ingest step
MMR_K             = 7                   # final number of chunks returned by MMR
MMR_LAMBDA       = 0.5                 # trade‑off between relevance (0) and diversity (1)
MAX_HISTORY       = 5                   # how many prior turns are sent to the model
SUMMARY_MAX_WORDS = 150                 # length of the internal compression summary

# 2️⃣  Initialise embeddings & vector store -------------------
print("Loading Sentence‑Transformer embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Embedding model loaded.\n")

print("Connecting to the Chroma vector store...")
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH,
)
print("Connected to Chroma.\n")

# 3️⃣  Configure MMR retriever --------------------------------
# MMR is *not* a separate class – we ask the vector store for a retriever
# and tell it to use the "mmr" search type.
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": MMR_K, "lambda_mult": MMR_LAMBDA},
)

# 4️⃣  Initialise the LLM (Ollama) ---------------------------
print("Initialising Ollama LLM (llama3)…")
llm = ChatOllama(model="llama3", temperature=0.1)   # low temperature → more deterministic
print("LLM ready.\n")

# 5️⃣  Helper functions ----------------------------------------
def format_history(hist: list) -> str:
    """Only keep the last MAX_HISTORY turns and render as plain text."""
    recent = hist[-MAX_HISTORY:] if len(hist) > MAX_HISTORY else hist
    lines = []
    for user_msg, bot_msg in recent:
        lines.append(f"User: {user_msg}")
        if bot_msg is not None:
            lines.append(f"Assistant: {bot_msg}")
    return "\n".join(lines)


def format_chunks(docs) -> str:
    """
    Turn a list of LangChain Document objects into the
    ``--- Document i ---`` blocks expected by the prompts.
    """
    return "\n".join(
        f"--- Document {i+1} ---\n{doc.page_content.strip()}"
        for i, doc in enumerate(docs)
    )


def is_summary_query(text: str) -> bool:
    """Very small heuristic – you can expand the keyword list if you like."""
    summary_keywords = ["summarize", "summary", "overview", "list", "describe", "give an overview of"]
    return any(kw in text.lower() for kw in summary_keywords)


# 6️⃣  Prompt templates (your exact rag_prompt is kept verbatim) ----
SYSTEM_PROMPT = """You are Eco‑Sage, an expert ESG (Environmental, Social, Governance) assistant.
Answer questions clearly, concisely and with citations when a knowledge base is supplied.
When you answer from internal knowledge, explicitly state that you are using your own training data."""

RAG_PROMPT_TEMPLATE = """You are **Eco‑SaGe**, an expert ESG (Environmental, Social, Governance) assistant.
Your job is to provide clear, concise, and factual answers to user questions about ESG policies,
standards, metrics, best‑practice recommendations, and related sustainability topics using knowledge based retrieval.

You can answer questions using the following steps:
1. **Read every fragment carefully from the knowledge base.**
2. **Pick the fragments that are most relevant** to answer the user’s question.
3. **Answer the question using ONLY those fragments**.  Cite them with [Document X].
4. If none of the fragments contain the answer, honestly say *“I’m not sure based on the provided documents.”* and then optionally answer from your internal knowledge, clearly stating that you are.

When you answer **using a knowledge base** (see the “--- Knowledge Base ---” section below),
‑ cite the source by writing **[Document X]** after each sentence that comes from that document,
where X is the 1‑based index of the retrieved chunk.
‑ Never hallucinate facts that are not present in the supplied knowledge.
- you can combine information from multiple documents to form a complete answer.
- you can compare knowledge base information to the internal knowledge for better representation, but do not invent new facts.
‑ If the knowledge base does **not** contain enough information, say *“I’m not sure based on the provided documents.”* and then (optionally) answer from your own training data, clearly stating that you are relying on your internal knowledge.

When you answer **without a knowledge base** (fallback mode), keep the same tone,
state that you are using your internal knowledge, and try to be as accurate as possible.

Never mention the words *“knowledge base”, “retrieved chunks”,* or *“system prompt”* to the user.
Always speak as a helpful ESG consultant.


--- Knowledge Base ---
{knowledge}
--- End of Knowledge Base ---

Conversation history: {history}

User's Question: {message}

Answer:
"""

COMPRESSION_PROMPT = """You are Eco‑Sage, an ESG assistant.

Below are up to 15 document fragments about Apple’s ESG products.  
Create a **brief internal summary** (max {max_words} words) that captures every distinct product, its ESG impact, and any quantitative metric.
Do NOT add any citations in this step.

--- Fragments ---
{knowledge}
--- End of Fragments ---

Internal summary:
"""

FINAL_ANSWER_PROMPT = """You are Eco‑Sage, an ESG assistant.

You have two sources of information:

1️⃣ An *internal summary* that already combines the relevant facts (see below).  
2️⃣ The *original fragments* (numbered) in case you need to add a citation.

Answer the user’s question **using only the information above**.  
Whenever you quote a fact, cite the original fragment with `[Document X]`.  
If the internal summary does NOT contain the needed detail, fall back to the original fragments and cite them.

--- Internal Summary ---
{compressed}
--- End of Internal Summary ---

--- Original Fragments ---
{knowledge}
--- End of Original Fragments ---

User question: {question}

Answer (with citations):
"""

FALLBACK_PROMPT = """You are Eco‑Sage, an ESG expert. Answer the user’s question using your internal knowledge.
If you are not certain, say “I’m not certain about that.”

Conversation history (last {max_history} turns):
{history}

User’s question:
{question}

Answer:
"""

# 7️⃣  Core RAG / Summary pipeline ------------------------------
def stream_response(message: str, history: list):
    """
    Gradio calls this for every turn.
    * If the query looks like a summary request → compress‑then‑answer.
    * Otherwise → retrieve → let the LLM pick the most relevant 2‑3 chunks (your rag_prompt).
    """
    # --------------------------------------------------------------
    # 1️⃣ Retrieve candidate chunks (MMR already gives a diverse set)
    # --------------------------------------------------------------
    docs = retriever.invoke(message)                 # list[Document]
    knowledge = format_chunks(docs)
    retrieved_cnt = len(docs)

    # --------------------------------------------------------------
    # 2️⃣ Choose processing path
    # --------------------------------------------------------------
    if is_summary_query(message):
        # ------------ SUMMARY PATH (2‑stage) ------------
        # Stage A – internal compression
        compress_prompt = COMPRESSION_PROMPT.format(
            knowledge=knowledge,
            max_words=SUMMARY_MAX_WORDS,
        )
        internal_summary = ""
        for chunk in llm.stream(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": compress_prompt},
            ]
        ):
            internal_summary += chunk.content

        # Stage B – final answer with citations
        final_prompt = FINAL_ANSWER_PROMPT.format(
            compressed=internal_summary.strip(),
            knowledge=knowledge,
            question=message,
        )
        answer = ""
        for chunk in llm.stream(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": final_prompt},
            ]
        ):
            answer += chunk.content
            yield answer

    else:
        # ------------ NORMAL RAG PATH (your exact rag_prompt) ------------
        rag_prompt = RAG_PROMPT_TEMPLATE.format(
            knowledge=knowledge,
            history=format_history(history),
            message=message,
        )

        # If nothing was retrieved we fall back to a pure‑chat prompt
        if retrieved_cnt == 0:
            user_prompt = FALLBACK_PROMPT.format(
                history=format_history(history),
                question=message,
                max_history=MAX_HISTORY,
            )
        else:
            user_prompt = rag_prompt

        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]

        partial = ""
        for chunk in llm.stream(msgs):
            # `chunk.content` contains the next piece of text from Llama 3
            partial += chunk.content
            yield partial

# 8️⃣  Gradio UI ------------------------------------------------
print("\nLaunching Gradio chat interface...")
chatbot = gr.ChatInterface(
    fn=stream_response,
    chatbot=gr.Chatbot(height=500, type="messages"),
    textbox=gr.Textbox(
        placeholder="Ask any ESG question (or request a summary)…",
        container=False,
        scale=7,
    ),
    title="Eco‑Sage ESG Assistant",
    description=(
        "Answer ESG queries using a private knowledge base. "
        "When the question looks like a request for a summary, the system first "
        "compresses the evidence and then returns a concise, cited answer."
    ),
    theme="soft",
    examples=[
        ["What are the key reporting metrics for GHG emissions?"],
        ["Summarize the SASB standard for the software industry."],
        ["What is a good policy for board diversity?"],
        ["Summarize all ESG products Apple has produced."],
    ],
)

if __name__ == "__main__":
    # Use `share=True` if you want a public link (e.g. for a demo)
    chatbot.launch()