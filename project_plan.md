

1. Integrate NLP learning functions in  into the RAG in for calling
2. Add more files into data (from 3 companies to 100 companies from sp500)
3. add sector data from  to RAG in chatbotv2


















# ESG Chatbot Project Plan & Workload Distribution

## 1. Project Overview
**Goal**: Build a specialized RAG-based ESG Chatbot that not only retrieves information from sustainability reports but also analyzes policies using custom ML models (XGBoost) and benchmarks companies against S&P 500 sector data.

**Core Architecture**:
1.  **Knowledge Base**: Vector DB (Chroma) storing 100+ company reports & frameworks.
2.  **Analysis Engine**: Custom XGBoost model (`score_predictions_py.py`) and Sector Analysis (`esg-risk-analysis...ipynb`).
3.  **Conversational Layer**: LLM (Llama3 via Ollama) orchestrating retrieval and scoring (`chatbotv2.py`).

---

## 2. Workload Distribution (3 Persons)

### 👤 Person A: Data Engineer & Knowledge Base Architect
**Focus**: Data Collection, Cleaning, Vector Database, and Scalability.
*Responsible for "The Library" and scaling to 100 companies.*

*   **[Task 2] Scale Data Pipeline**:
    *   **Objective**: Move from 3 demo companies to 100 S&P 500 companies.
    *   **Action**: Automate the PDF downloading/scraping process if possible, or organize the manual collection of 100+ PDFs.
    *   **Action**: Update `ingestfiles.py` (implied file) to handle batch processing of large numbers of files without crashing (implement checkpoints or incremental indexing).
*   **[Task 3] Integrate Sector Data**:
    *   **Objective**: Make the Kaggle dataset insights available to the bot.
    *   **Action**: Extract the cleaned sector averages and risk percentiles from `esg-risk-analysis-insights-from-s-p-500-companies.ipynb`.
    *   **Action**: Create a structured summary of this data (e.g., a JSON or Markdown file "sector_benchmarks.md") and ingest it into the Vector Database so the bot can retrieve "Average Energy Sector Risk Score" when asked.
*   **[Task 4 - New] Metadata Enhancement**:
    *   **Objective**: Improve retrieval accuracy.
    *   **Action**: Modify the chunking logic to attach metadata to every chunk (e.g., `{"company": "Apple", "sector": "Technology", "year": "2023"}`). This allows the bot to filter searches (e.g., "Compare Apple to other Tech companies").

### 👤 Person B: ML Engineer & Analyst
**Focus**: Predictive Modeling, Scoring Functions, and Quantitative Analysis.
*Responsible for "The Analysis Engine" and `score_predictions_py.py`.*

*   **[Task 1] Refine NLP/XGBoost Model**:
    *   **Objective**: Improve the `score_predictions_py.py` model.
    *   **Action**: The current script trains a model. You need to create a **clean inference function** that `chatbotv2.py` can call.
    *   **Deliverable**: A function `predict_esg_score(policy_text)` that loads the saved XGBoost model and returns the 4 risk scores.
*   **[Task 1] Model Integration Preparation**:
    *   **Objective**: Ensure the model is robust.
    *   **Action**: Validate the model against the new 100-company dataset (provided by Person A). Retrain the XGBoost model with this larger dataset to improve accuracy.
*   **[Task 4 - New] Keyword/Topic Classifier**:
    *   **Objective**: Route user queries correctly.
    *   **Action**: As suggested in the README, build a small classifier or keyword extractor to determine if a user is asking for a *score* (use XGBoost) or a *summary* (use LLM).

### 👤 Person C: Chatbot Architect & Integrator
**Focus**: Application Logic, RAG Integration, and User Experience.
*Responsible for "The Spokesperson" and `chatbotv2.py`.*

*   **[Task 1] Integrate ML Predictions into Chatbot**:
    *   **Objective**: Connect Person B's model to the chat loop.
    *   **Action**: Modify `chatbotv2.py`. When the user provides a policy text, add a tool/function call to `predict_esg_score()`.
    *   **Action**: Update the System Prompt to explain how to interpret these scores (e.g., "The model predicts a Governance Risk of 5.2, which is low/good").
*   **[Task 3] RAG Retrieval Logic**:
    *   **Objective**: Fetch specific sector data.
    *   **Action**: Enhance the retrieval in `chatbotv2.py`. If the user asks for a benchmark, ensure the retrieval query targets the "sector_benchmarks" documents created by Person A.
*   **[Task 4 - New] UI/UX Improvements**:
    *   **Objective**: Make the Gradio interface professional.
    *   **Action**: Add UI elements to display the numerical scores visually (e.g., a progress bar or chart) alongside the text response.

---

## 3. Collaborative Roadmap

| Phase | Person A (Data) | Person B (ML) | Person C (Chatbot) |
| :--- | :--- | :--- | :--- |
| **Week 1** | Collect 100 PDFs. Clean Sector Data from Notebook. | Refactor `score_predictions.py` into a reusable `inference.py` module. | Study `chatbotv2.py`. Design the "Tool Calling" logic for the LLM. |
| **Week 2** | Ingest 100 PDFs + Sector Data into ChromaDB. | Retrain XGBoost on 100 companies. Validate accuracy. | Integrate `inference.py` into Chatbot. Test "Scoring" flow. |
| **Week 3** | Optimize Vector Search (add metadata filters). | Experiment with a Topic Classifier (Score vs. Search). | Improve System Prompts. Add UI visualizations for scores. |

## 4. Immediate Next Steps
1.  **Person A**: Run `ingestfiles.py` (check if it exists/needs creation) on the current small dataset to ensure the pipeline works before scaling.
2.  **Person B**: Create a new file `prediction_service.py` that imports the trained model and exposes a simple `predict(text)` function.
3.  **Person C**: Update `chatbotv2.py` to import `prediction_service.py` and test calling it with a dummy string.
