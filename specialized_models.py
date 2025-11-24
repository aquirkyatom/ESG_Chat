# ==================== specialized_models.py ====================
import joblib
import numpy as np
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# 1. Configuration
MODEL_PATH = 'data/esg_multi_output_model.pkl'
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# 2. Global Loader
print("Loading XGBoost Model, Embeddings, and Expansion LLM...")
try:
    xgb_model = joblib.load(MODEL_PATH)
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    # Initialize a specific instance of Llama3 just for expansion
    # Temp 0.4 allows for creative writing while staying factual
    llm_expander = ChatOllama(model="llama3", temperature=0.4)
    
    MODEL_LOADED = True
except Exception as e:
    print(f"CRITICAL ERROR: Could not load model/embeddings: {e}")
    MODEL_LOADED = False

# 3. Clean Text (EXACT copy from training script)
def clean_text(text):
    if not isinstance(text, str): return ""
    text = str(text).lower()
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\[\*\*.*?\*\*\]', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def format_prediction(scores, source_type):
    """Helper to make the output readable for the LLM"""
    # scores = [Total, Env, Soc, Gov]
    return (
        f"\n--- 🤖 AI RISK MODEL ANALYSIS ({source_type}) ---\n"
        f"Total ESG Risk:      {scores[0]:.2f}\n"
        f"  - Environment:     {scores[1]:.2f}\n"
        f"  - Social:          {scores[2]:.2f}\n"
        f"  - Governance:      {scores[3]:.2f}\n"
        f"--------------------------------------------\n"
    )

# --- FUNCTION 1: FOR COMPANY ANALYSIS (Uses RAG Docs) ---
# No expansion needed here because RAG documents are already 
# formal reports (PDFs) similar to training data.
def predict_company_risk(rag_documents: list) -> str:
    if not MODEL_LOADED or not rag_documents:
        return "Error: Model not loaded or no documents found."

    # 1. Clean & Embed each retrieved chunk
    cleaned_texts = [clean_text(d) for d in rag_documents if len(clean_text(d)) > 15]
    if not cleaned_texts:
        return "Retrieved documents contained insufficient text."

    vectors = np.array(embedder.embed_documents(cleaned_texts))
    
    # 2. Predict on every chunk and Average
    # We take the consensus of the retrieved evidence.
    preds = xgb_model.predict(vectors)
    avg_scores = np.mean(preds, axis=0)

    return format_prediction(avg_scores, "Company Analysis Consensus")

# --- FUNCTION 2: FOR POLICY TESTING (Uses User Input) ---
# UPDATED: Uses LLM Expansion to handle short inputs
def predict_policy_risk(user_policy_text: str) -> str:
    if not MODEL_LOADED:
        return "Error: Model not loaded."
    
    if len(user_policy_text.strip()) < 3:
         return "Error: Input text is too short."

    # --- STEP 1: EXPANSION ---
    # Convert short user input (OOD) into formal report text (In-Distribution)
    print(f"DEBUG: Expanding user input: '{user_policy_text}'")
    
    expansion_prompt = f"""
    You are an ESG Report Writer.
    Expand the following short policy statement into a formal, detailed paragraph (approx 80-100 words) as if it appeared in an annual ESG Risk Report.
    Use professional terminology (e.g., 'compliance', 'emissions', 'governance structure', 'audit', 'regulatory impact').
    Do not be conversational. Output only the report text.
    
    Policy Statement: "{user_policy_text}"
    
    Formal Report Fragment:
    """
    
    try:
        # Generate the expanded text
        expanded_text = llm_expander.invoke(expansion_prompt).content
    except Exception as e:
        return f"Error during text expansion: {str(e)}"

    # --- STEP 2: PREDICTION ---
    # Now we feed the EXPANDED text to the model
    cleaned = clean_text(expanded_text)
    
    # Embed single document (wrap in list)
    vector = np.array(embedder.embed_documents([cleaned]))
    
    # Predict (Single instance)
    pred = xgb_model.predict(vector)[0] 
    
    return format_prediction(pred, "Hypothetical Policy Evaluation (Expanded Context)")