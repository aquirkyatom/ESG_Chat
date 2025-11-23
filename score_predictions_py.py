import pandas as pd
import numpy as np
import os
import joblib
import re
import csv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import SpacyTextSplitter
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

# ==========================================
# CONFIGURATION
# ==========================================
CSV_PATH = r"data/SP 500 ESG Risk Ratings.csv"
PDF_FOLDER = r"data/esg reports sp500"
MODEL_OUTPUT_FILE = 'data/esg_multi_output_model.pkl'
TRAINING_DATA_CSV = 'data/processed_esg_training_data.csv'
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

PAGES_TO_READ = 10 
MAX_CHUNKS_PER_COMPANY = 60 

TARGET_COLS = [
    'Total ESG Risk score', 
    'Environment Risk Score', 
    'Social Risk Score', 
    'Governance Risk Score'
]

# ==========================================
# FUNCTION 1: CLEAN TEXT
# ==========================================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = str(text).lower()
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\[\*\*.*?\*\*\]', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==========================================
# FUNCTION 2: FIX TICKER
# ==========================================
def get_clean_ticker(identifier_string):
    """
    Removes trailing digits (e.g. AAPL1 -> AAPL).
    """
    s = str(identifier_string).strip().upper()
    clean_s = re.sub(r'\d+$', '', s)
    return clean_s

# ==========================================
# FUNCTION 3: TRANSFORM TO HORIZONTAL
# ==========================================
def transform_to_horizontal(df, score_cols):
    print("--- Pivoting Data to Horizontal Format ---")
    group_cols = ['Symbol'] + score_cols
    grouped = df.groupby(group_cols)['Text Chunk'].apply(list).reset_index()
    text_columns = pd.DataFrame(grouped['Text Chunk'].tolist(), index=grouped.index)
    text_columns.columns = [f'Text Chunk {i+1}' for i in range(text_columns.shape[1])]
    final_df = pd.concat([grouped.drop(columns=['Text Chunk']), text_columns], axis=1)
    return final_df

# ==========================================
# 1. DATA PREPARATION
# ==========================================
print("\n--- 1. Loading and Cleaning Data ---")
os.makedirs(os.path.dirname(MODEL_OUTPUT_FILE), exist_ok=True)
os.makedirs(os.path.dirname(TRAINING_DATA_CSV), exist_ok=True)

df = pd.read_csv(CSV_PATH)

missing_cols = [c for c in TARGET_COLS if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

df = df.dropna(subset=TARGET_COLS)
for col in TARGET_COLS:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=TARGET_COLS)

# Clean Tickers
df['Symbol'] = df['Symbol'].apply(get_clean_ticker)

# --- NEW STRICT FILE MATCHING LOGIC ---
all_pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
ticker_to_files = {}
valid_tickers = []

print("Mapping files to tickers (Strict Match)...")

for ticker in df['Symbol'].unique():
    matches = []
    
    # Create a Regex Pattern for STRICT matching
    # ^       = Start of string
    # ticker  = The ticker string (e.g., "A")
    # (\W|$)  = Must be followed by a non-letter (like . or _) OR end of string
    # This ensures "A" matches "A.pdf" or "A_Report.pdf" BUT NOT "AAPL.pdf"
    pattern = re.compile(rf"^{re.escape(str(ticker))}(\W|_|\d|$)", re.IGNORECASE)
    
    for fname in all_pdf_files:
        if pattern.match(fname):
            matches.append(fname)
            
    if matches:
        ticker_to_files[ticker] = matches
        valid_tickers.append(ticker)
        # Debug print to verify correct matching
        if ticker in ["A", "AAPL", "MS", "MSFT"]: 
            print(f"  -> Matched {ticker}: {matches}")

df_final = df[df['Symbol'].isin(valid_tickers)].copy()
print(f"Final Training Set: {len(df_final)} companies.")

score_map = {}
for index, row in df_final.iterrows():
    score_map[row['Symbol']] = row[TARGET_COLS].values.tolist()

# ==========================================
# 2. PROCESSING
# ==========================================
print(f"\n--- 2. Processing First {PAGES_TO_READ} Pages of PDFs ---")

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
text_splitter = SpacyTextSplitter(chunk_size=1000, chunk_overlap=200, pipeline="en_core_web_lg")

X = [] 
y = [] 
csv_export_rows = [] 

for index, row in df_final.iterrows():
    ticker = row['Symbol']
    target_scores = score_map[ticker] 
    files = ticker_to_files[ticker]
    
    print(f"Processing {ticker}... ", end="")
    
    company_chunks = []
    for filename in files:
        file_path = os.path.join(PDF_FOLDER, filename)
        try:
            loader = PyPDFLoader(file_path)
            all_pages = loader.load()
            selected_pages = all_pages[:PAGES_TO_READ]
            company_chunks.extend(text_splitter.split_documents(selected_pages))
        except Exception as e:
            print(f"(Error reading {filename})", end=" ")

    if MAX_CHUNKS_PER_COMPANY and len(company_chunks) > MAX_CHUNKS_PER_COMPANY:
        company_chunks = company_chunks[:MAX_CHUNKS_PER_COMPANY]
    
    if not company_chunks:
        print("Empty.")
        continue

    final_cleaned_texts = []
    for doc in company_chunks:
        cleaned_text = clean_text(doc.page_content)
        if len(cleaned_text) > 15:
            final_cleaned_texts.append(cleaned_text)
    
    if not final_cleaned_texts: 
        print("No valid text.")
        continue

    # A. Training Data
    vectors = embedding_model.embed_documents(final_cleaned_texts)
    X.extend(vectors)
    y.extend([target_scores] * len(vectors))

    # B. CSV Data
    for text_snippet in final_cleaned_texts:
        row_data = {
            'Symbol': ticker,
            'Text Chunk': text_snippet,
            'Total ESG Risk score': target_scores[0],
            'Environment Risk Score': target_scores[1],
            'Social Risk Score': target_scores[2],
            'Governance Risk Score': target_scores[3]
        }
        csv_export_rows.append(row_data)
    
    print(f"Done ({len(vectors)} chunks).")

# ==========================================
# 3. SAVE CSV
# ==========================================
print(f"\n--- 3. Transforming and Saving CSV ---")

if csv_export_rows:
    df_vertical = pd.DataFrame(csv_export_rows)
    df_horizontal = transform_to_horizontal(df_vertical, score_cols=TARGET_COLS)
    
    df_horizontal.to_csv(
        TRAINING_DATA_CSV, 
        index=False, 
        encoding='utf-8-sig', 
        quoting=csv.QUOTE_ALL
    )
    print(f"Successfully created '{TRAINING_DATA_CSV}'")
    print(f"Rows: {len(df_horizontal)}")
else:
    print("No data processed.")

# ==========================================
# 4. TRAINING
# ==========================================
print(f"\n--- 4. Training Multi-Output XGBoost ---")

if len(X) > 0:
    X_np = np.array(X)
    y_np = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.2, random_state=42)

    xgb_estimator = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        n_jobs=-1
    )

    model = MultiOutputRegressor(xgb_estimator)
    model.fit(X_train, y_train)

    # Evaluationn.")
    predictions = model.predict(X_test)
    print("\n--- Model Accuracy (RMSE) ---")
    for i, col_name in enumerate(TARGET_COLS):
        rmse = root_mean_squared_error(y_test[:, i], predictions[:, i])
        print(f"{col_name} Error: +/- {rmse:.2f}")

    joblib.dump(model, MODEL_OUTPUT_FILE)
    print(f"\nModel saved to: {MODEL_OUTPUT_FILE}")
else:
    print("Not enough data to train")