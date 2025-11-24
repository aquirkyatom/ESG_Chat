import gradio as gr
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

# Import your Bridge (Make sure you updated specialized_models.py with the Expansion logic!)
import specialized_models 

# ==================== CONFIGURATION ====================
CHROMA_PATH = r"chroma_db"
COLLECTION_NAME = "esg_collection"
MMR_K = 7              
NUM_CANDIDATES = 2     
MAX_RETRIES = 3        

# ==================== SETUP ====================
print("Loading Models...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH,
)
retriever = vector_store.as_retriever(search_kwargs={"k": MMR_K})
llm = ChatOllama(model="llama3", temperature=0.1) # Keep temp low for strictness
print("System Ready.")

# ==================== PROMPTS (THE GUARDRAILS) ====================

# --- 1. THE ROUTER ---
router_prompt = ChatPromptTemplate.from_template(
    """Analyze the user's request and output JSON ONLY.
    
    1. If user asks to SUMMARIZE/OVERVIEW a framework or policy:
       return {{"tool": "SUMMARIZER", "topic": "topic name"}}
       
    2. KEY RULE: If the user implies TESTING/SCORING a text input 
       (e.g., "Score this: [text]", "Evaluate: [text]", "We use 100% coal"):
       return {{"tool": "POLICY_TESTER", "policy_text": "extracted text"}}
       
    3. If user asks to CALCULATE/SCORE a specific COMPANY (e.g., "Risk score for Apple"):
       return {{"tool": "COMPANY_SCORER", "company_name": "company name"}}
    
    4. If the user asks "Why?", "Explain", or refers to previous chat:
       return {{"tool": "CONVERSATIONAL"}}

    5. Default:
       return {{"tool": "STANDARD_RAG"}}

    Question: {question}
    """
)

# --- 2. THE RISK ANALYST (STRICT ANTI-HALLUCINATION) ---
# This is the firewall. It prevents the LLM from making up numbers.
risk_prompt = ChatPromptTemplate.from_template(
    """You are an ESG Risk Consultant. 
    You have received data from an internal AI Calculation Engine.
    
    Data Source:
    {context}
    
    Task: Present the risk assessment to the user.
    
    STRICT RULES (DO NOT IGNORE):
    1. Look for the section "--- 🤖 AI RISK MODEL ANALYSIS ---".
    2. If that section exists, copy the numbers (Total ESG risk score, Environment risk score, Social risk score, Governance risk score) EXACTLY.
    3. DO NOT generate your own numbers.
    5. If the "AI RISK MODEL ANALYSIS" section is MISSING, you MUST reply:
       "Error: I could not connect to the risk model to score this input."
    
    Question: {question}
    Answer:"""
)

# --- 3. THE SUMMARIZER ---
summary_prompt = ChatPromptTemplate.from_template(
    """You are an ESG Expert.
    Context: {context}
    Task: Summary for {topic}
    
    STRICT FORMAT:
    ## 1. Core Objectives
    [Content]
    ## 2. Key Standards
    [Content]
    ## 3. Critical Metrics
    [Content]
    """
)

# --- 4. GENERAL QA ---
qa_prompt = ChatPromptTemplate.from_template(
    """Answer using ONLY the provided context.
    Context: {context}
    Question: {question}
    Answer:"""
)

# --- 5. VOTER ---
voting_prompt = ChatPromptTemplate.from_template(
    """Select the better response (1 or 2).
    Candidate 1: {cand1}
    Candidate 2: {cand2}
    Return "1" or "2"."""
)

# ==================== LOGIC ====================

def validate_summary(text: str) -> bool:
    return "## 1. Core Objectives" in text

def format_history(hist):
    if not hist: return "No history."
    return "\n".join([f"User: {h[0]}\nAssistant: {h[1]}" for h in hist[-3:]])

def flow_coordinator(message: str, history: list):
    current_state = {"context": "", "candidates": []}

    # --- NODE 1: ROUTING ---
    try:
        route = (router_prompt | llm | JsonOutputParser()).invoke({"question": message})
        tool = route.get("tool", "STANDARD_RAG")
    except:
        tool = "STANDARD_RAG"
        route = {}
    
    print(f"DEBUG: Selected Tool -> {tool}")

    # --- NODE 2: EXECUTION ---
    
    if tool == "POLICY_TESTER":
        txt = route.get("policy_text", message)
        # This calls your specialized model (which now uses Expansion)
        model_out = specialized_models.predict_policy_risk(txt)
        
        current_state["context"] = f"User Input: {txt}\n\n{model_out}"
        active_prompt = risk_prompt 
        validation_fn = lambda x: True

    elif tool == "COMPANY_SCORER":
        company = route.get("company_name", message)
        docs = retriever.invoke(company)
        
        if not docs:
            yield "I could not find data for that company."
            return

        raw_texts = [d.page_content for d in docs]
        model_out = specialized_models.predict_company_risk(raw_texts)
        doc_text = "\n".join([d.page_content[:200] for d in docs])
        
        current_state["context"] = f"{model_out}\n\nReference Docs:\n{doc_text}"
        active_prompt = risk_prompt
        validation_fn = lambda x: True

    elif tool == "SUMMARIZER":
        topic = route.get("topic", message)
        docs = retriever.invoke(topic)
        current_state["context"] = "\n".join([d.page_content for d in docs])
        active_prompt = summary_prompt 
        validation_fn = validate_summary

    elif tool == "CONVERSATIONAL":
        hist_text = format_history(history)
        current_state["context"] = f"HISTORY:\n{hist_text}"
        active_prompt = qa_prompt
        validation_fn = lambda x: True

    else: # STANDARD RAG
        docs = retriever.invoke(message)
        current_state["context"] = "\n".join([d.page_content for d in docs])
        active_prompt = qa_prompt
        validation_fn = lambda x: True

    # --- NODE 3: GENERATION ---
    chain = active_prompt | llm | StrOutputParser()
    topic = route.get("topic", message)

    for i in range(NUM_CANDIDATES):
        valid = False
        attempts = 0
        while not valid and attempts < MAX_RETRIES:
            if tool == "SUMMARIZER":
                inputs = {"context": current_state["context"], "topic": topic}
            else:
                inputs = {"context": current_state["context"], "question": message}

            response = chain.invoke(inputs)

            # Extra check for hallucination in Risk Scenarios
            if tool in ["POLICY_TESTER", "COMPANY_SCORER"]:
                if "Error:" in response or "Environment: 10" in response:
                    # Treat vague/bad answers as failures to force retry
                    valid = False 
                else:
                    valid = True
            elif validation_fn(response):
                valid = True
            
            if valid:
                current_state["candidates"].append(response)
            else:
                attempts += 1
                print(f"Branch {i}: Retrying...")

        if not valid:
             current_state["candidates"].append("I'm sorry, I couldn't generate a valid calculation.")

    # --- NODE 4: VOTING ---
    if len(current_state["candidates"]) > 1:
        vote = (voting_prompt | llm | StrOutputParser()).invoke({
            "cand1": current_state["candidates"][0],
            "cand2": current_state["candidates"][1]
        })
        final_answer = current_state["candidates"][0] if "1" in vote else current_state["candidates"][1]
    else:
        final_answer = current_state["candidates"][0]

    yield final_answer

# ==================== UI ====================
chatbot = gr.ChatInterface(
    fn=flow_coordinator,
    chatbot=gr.Chatbot(height=650),
    title="ESG Agent (Anti-Hallucination Edition)",
    description="Ask to score companies or policies.",
)

if __name__ == "__main__":
    chatbot.launch()