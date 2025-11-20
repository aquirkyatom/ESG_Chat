# ESG_Chat
Chatbot for APAI 4011

Current plans, **gen by AI** a RAG model

data folders please see dropbox

I have built a python code to store the data, so my plan now is to build from a existing chatbot model, than we use train our own functions/code for machine learning, than use rag to call those functions when required to make a bot

## Data ##
 Frameworks: Frameworks from https://www.hkex.com.hk/Listing/Sustainability/ESG-Academy/External-References/ESG-Frameworks?sc_lang=en
*     GRI 2023 framework
*     SASB Technology & Communications -> Software & IT Services:
           ** Financials -> Commercial Banks
                        ** Health Care -> Biotechnology & Pharmaceuticals
                         ** Extractives & Minerals Processing -> Oil & Gas – Exploration & Production
                 ** Food & Beverage -> Agricultural Products
            **  Infrastructure -> Electric Utilities & Power Generators
           ** Consumer Goods -> Multiline and Specialty Retailers & Distributors
           **  Transportation -> Automobiles

* TCFD 2017 report https://assets.bbhub.io/company/sites/60/2021/10/FINAL-2017-TCFD-Report.pdf
* SDG report https://unstats.un.org/sdgs/report/2025/The-Sustainable-Development-Goals-Report-2025.pdf
* ISSB report https://www.ifrs.org/content/dam/ifrs/publications/pdf-standards-issb/english/2023/issued/part-a/issb-2023-a-ifrs-s1-general-requirements-for-disclosure-of-sustainability-related-financial-information.pdf?bypass=on https://www.ifrs.org/content/dam/ifrs/publications/pdf-standards-issb/english/2023/issued/part-a/issb-2023-a-ifrs-s2-climate-related-disclosures.pdf?bypass=on
kaggle datasets

  




























### Recommended Model: A Multi-Component RAG System

Think of your chatbot not as a single brain, but as a system with three specialized parts working together:

1.  **The Knowledge Base (Vector Database):** The "Library"
2.  **The Analysis Engine (Fine-tuned Models):** The "Specialist Analyst"
3.  **The Conversational Layer (Large Language Model):** The "Spokesperson"

This is a state-of-the-art approach for building specialized, knowledge-intensive bots.

#### 1. The Knowledge Base: Your ESG Universe

This is the foundation. Your bot can't benchmark a policy against data it hasn't read. This component is a database of all relevant ESG information.

*   **What it is:** A **Vector Database** (e.g., Pinecone, Weaviate, or a local one like ChromaDB).
*   **How it's built:**
    1.  **Gather Documents:** You collect a massive corpus of ESG documents:
        *   **Sustainability Reports:** From hundreds of companies across different industries (this is where your idea is perfect).
        *   **ESG Frameworks:** The complete standards from GRI, SASB, TCFD, etc.
        *   **Rating Agency Criteria:** Methodologies from MSCI, Sustainalytics (if you can find them).
        *   **Regulations:** Key government regulations on ESG topics.
    2.  **Chunk and Vectorize:** You split these documents into small, meaningful chunks (e.g., paragraphs). You then use a **Sentence-BERT** model (like `all-MiniLM-L6-v2`) to convert each chunk into a numerical vector (an "embedding"). This vector represents the semantic meaning of the text.
    3.  **Store:** You store these vectors in the vector database, which is optimized for incredibly fast "similarity search."

#### 2. The Analysis Engine: The "Scoring" Brain

This is where the real "magic" happens. When a user submits a policy, this engine analyzes it. It's not one model, but a pipeline of a few specialized, fine-tuned models.

*   **Component A: Policy Topic Classifier (Fine-tuned BERT)**
    *   **Model:** A lightweight BERT model like `DistilBERT` or `RoBERTa`, fine-tuned for multi-label text classification.
    *   **Task:** When it receives a user's policy, its job is to classify it. For example:
        *   **Input:** "Our company will reduce Scope 1 emissions by 30% by 2030."
        *   **Output:** `['Environment', 'GHG Emissions', 'Forward-Looking Statement']`
    *   **Training:** You need a labeled dataset of policy snippets and their corresponding topics. You can create this from your sustainability reports.

*   **Component B: Benchmark Retriever (Similarity Search)**
    *   **Model:** This isn't a trained model, but an algorithm. It's a query to your vector database.
    *   **Task:**
        1.  It takes the user's policy and its topics from the classifier.
        2.  It queries the vector database to find the **most similar policy chunks** from your knowledge base (e.g., "find me the 5 most similar GHG emission policies from the Technology sector").
    *   **Output:** A ranked list of "best-in-class" examples from real companies or frameworks. This is your **"benchmark."**

*   **Component C (Optional, Advanced): Policy Strength Scorer**
    *   **Model:** A regression model (can be another fine-tuned BERT or even XGBoost).
    *   **Task:** This model attempts to assign a numerical **"point score"** to the policy.
        *   **Input:** The user's policy text (or its embedding).
        *   **Output:** A score from 1-10.
    *   **Training:** This is the hardest part. You need a large dataset where policies are already scored. You might need to create this yourself by hand or use ESG scores from rating agencies as a proxy for the labels.

#### 3. The Conversational Layer: The Chatbot Interface

This is the part the user actually talks to.

*   **Model:** A powerful, general-purpose **Large Language Model (LLM)** like GPT-4, Llama 3, Gemini, or a fine-tuned open-source equivalent.
*   **Task:** Its job is to be the "spokesperson" that orchestrates everything.
    1.  It receives the user's natural language query (e.g., "What do you think of this policy for our carbon emissions?").
    2.  It understands the user's *intent* is to get a benchmark. It extracts the policy text.
    3.  It calls the **Analysis Engine** (Step 2) with the policy text.
    4.  The Analysis Engine returns its findings (e.g., `Topics: ['GHG'], Benchmarks: [Example from Apple, Example from GRI Standard]`).
    5.  The LLM receives this structured data and **synthesizes it** into a helpful, human-readable paragraph, presenting the benchmark and explaining the key points.

---

### Comparison to Your Initial Idea & Development Plan

| Your Idea                               | Recommended Architecture Component                                   | Why it's an improvement                                                                                             |
| --------------------------------------- | -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Use BERT to process input               | **3. Conversational Layer (LLM)**                                    | A full LLM can handle conversation, intent recognition, and synthesizing results, which is more than just BERT processing. |
| ML model trained on ESG reports         | **1. Knowledge Base** & **2. Analysis Engine**                       | This breaks down the vague "ML model" into specific, trainable components (a classifier, a retriever, a scorer).      |
| Compare against Kaggle datasets         | **1. Knowledge Base** (as a data source)                             | Instead of a static comparison, this creates a dynamic, searchable knowledge base for real-time benchmarking.     |

### Your Step-by-Step Development Plan

1.  **Phase 1: Build the Knowledge Base (The Library).**
    *   Gather all your PDFs and text files (sustainability reports, GRI standards).
    *   Write scripts to parse, clean, and chunk the text.
    *   Use a Sentence-BERT model to vectorize every chunk.
    *   Load all vectors into a vector database like ChromaDB.

2.  **Phase 2: Build the Analysis Engine (The Analyst).**
    *   **Start with the Classifier.** Create a labeled dataset of policy snippets and their ESG topics. Fine-tune a `DistilBERT` model on this classification task. This is your most valuable component.
    *   **Implement the Retriever.** Write the code to query your vector database.

3.  **Phase 3: Build the Conversational Layer (The Spokesperson).**
    *   Choose an LLM (e.g., via the OpenAI API or a local model).
    *   Use a framework like **LangChain** or LlamaIndex to easily chain the steps together: `User Input -> LLM -> Analysis Engine -> LLM -> Final Output`.
    *   Focus on "prompt engineering" to tell the LLM how to use the retrieved benchmarks to answer the user's question.

This modular approach is complex, but it's the right way to build a powerful and accurate ESG chatbot. Your initial idea was the perfect starting point, and this architecture is the professional way to bring it to life.
