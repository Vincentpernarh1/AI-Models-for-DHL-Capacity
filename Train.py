import pandas as pd
import re
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List

# --- Configuration ---
# Path to the CSV file you just created.
CSV_FILE_PATH = "C:/Users/perna/Desktop/AI/training_data.csv"

# The folder where the final FAISS index will be saved.
FAISS_INDEX_PATH = "company_faiss_index"

# The embedding model to use (must match your main application).
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def clean_text(text: str) -> str:
    """Cleans a string by removing invalid characters and extra whitespace."""
    if not isinstance(text, str):
        text = str(text)
    try:
        text = re.sub(r'\s+', ' ', text).strip()
        text = ''.join(char for char in text if char.isprintable())
        return text
    except Exception:
        return ""

def load_and_format_csv_data(file_path: str) -> List[str]:
    """Loads data from a CSV file and formats each row into a structured string."""
    if not os.path.exists(file_path):
        print(f"⚠️  File not found: {file_path}")
        return []

    try:
        df = pd.read_csv(file_path, header=0, dtype=str).fillna("")

        if df.empty:
            print(f"⚠️  The file is empty: {file_path}")
            return []

        formatted_documents = []
        for _, row in df.iterrows():
            # Format each row as: "| Question: ... | Answer: ... |"
            row_text = " | ".join(
                [f"{clean_text(col)}: {clean_text(row[col])}" for col in df.columns if clean_text(row[col])]
            )
            if row_text:
                formatted_documents.append(f"| {row_text} |")

        print(f"✅ Loaded and formatted {len(formatted_documents)} documents from {file_path}")
        return formatted_documents

    except Exception as e:
        print(f"❌ Error processing CSV file {file_path}: {e}")
        return []

def create_faiss_index():
    """Main function to load all data, generate embeddings, and save the FAISS index."""
    print("🚀 Starting data processing...")

    documents = load_and_format_csv_data(CSV_FILE_PATH)

    if not documents:
        print("❌ No documents were loaded. Please check the CSV_FILE_PATH. Aborting.")
        return

    try:
        print(f"🔄 Initializing embedding model: {EMBEDDING_MODEL_NAME}...")
        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'}
        )
        print("✅ Embedding model loaded.")

        print("🔄 Building FAISS vector store... (This may take a moment)")
        vector_store = FAISS.from_texts(texts=documents, embedding=embedding_model)
        print("✅ Vector store built successfully.")

        vector_store.save_local(FAISS_INDEX_PATH)
        print(f"💾 FAISS index successfully saved to folder: '{FAISS_INDEX_PATH}'")

    except Exception as e:
        print(f"❌ An error occurred during FAISS index creation: {e}")

if __name__ == "__main__":
    create_faiss_index()
