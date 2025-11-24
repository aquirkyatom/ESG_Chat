
# 24/11/2025
###  `Chatbotv3.py` 
This file implements  **Agentic Flow Architecture**. Uses **Function Calling** and **Flow Engineering** to manage complex logic.

*   **Intelligent Routing (Function Calling):**
    *   Instead of treating every input as a search query, the **Router Node** analyzes user intent to select the correct tool.
    *   **Tools available:** `SUMMARIZER` (for frameworks), `POLICY_TESTER` (for user input), `COMPANY_SCORER` (for analysis), `CONVERSATIONAL` (for chat history), and `STANDARD_RAG`.
*   **Modularized Prompts:**
    *   Replaces the single generic prompt with specialized, task-specific instructions:
        *   `router_prompt`: For JSON-based decision making.
        *   `risk_prompt`: **Anti-Hallucination** prompt that strictly forbids guessing numbers.
        *   `summary_prompt`: Enforces strict Markdown headers (Objectives, Standards, Metrics).
*   **Self-Correction & Validation:**
    *   Implements a **Validation Loop**. If the LLM generates a summary with the wrong format or attempts to hallucinate a risk score without the model's data, the system wipes the answer and retries automatically.
*   **Voting Mechanism:**
    *   Generates multiple parallel responses (candidates) and uses a "Judge" prompt to select the most accurate one before showing it to the user.




###  `specialized_models.py` 
This file acts as the **Logic Adapter**, bridging the gap between unstructured text (LLM/User) and ML model.

*   **Model Integration:**
    *   Loads the pre-trained XGBoost model (`esg_multi_output_model.pkl`) and the Embedding Model (`all-MiniLM-L6-v2`) into memory once to ensure low latency.
*   **Synthetic Data Expansion (The "Translator"):**
    *   **Problem Solved:** The XGBoost model was trained on long, formal reports, but users type short sentences (e.g., "We use coal").
    *   **Solution:** It uses an internal LLM instance to **expand** short user inputs into formal, report-style paragraphs *before* sending them to the XGBoost model. This ensures the math remains accurate even for short inputs.
*   **Standardized Output:**
    *   Converts raw Numpy arrays into a formatted string (`--- 🤖 AI RISK MODEL ANALYSIS ---`) that the `Chatbotv3.py` prompts are trained to recognize and parse.

### Data Flow
1.  **User** asks a question in `Chatbotv3.py`.
2.  **Router** decides it needs a calculation and calls `specialized_models.py`.
3.  **Bridge** processes the text (expanding it if necessary), runs the `.pkl` model, and returns a string.
4.  **Chatbot** injects that string into the context, validates the final answer, and presents it to the user.

### updated `ingestfiles.py`
added a `clean_document_content` function and inserted a **Pre-processing Step** right before the text splitting.
1.  **Table of Contents Removal:** Uses Regex to detect lines that end in dots and numbers (e.g., `Environment ......... 15`).
2.  **Page Number Removal:** Removes lines that are just digits.
3.  **Header/Footer Noise:** Removes lines containing specific keywords like "Table of Contents" or "Index".
