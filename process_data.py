import pandas as pd
import re
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
 
# Load Excel Data with proper encoding and cleaning
def load_excel(file_path):
    try:
        df = pd.read_excel(file_path, header=0, dtype=str)  # Force all columns as strings
 
        if df.empty:
            print(f"⚠️ The file {file_path} is empty!")
            return []
 
        structured_data = []
        for _, row in df.iterrows():
            # Format row as "| Column1 = Value1 , Column2 = Value2 , ... |"
            row_text = "| " + "_".join(
                [f"_{col}:{clean_text(str(row[col]))}" for col in df.columns if pd.notna(row[col])]
            ) + " |"
            structured_data.append(row_text)  # Each row is stored as formatted text
 
        print(f"✅ Loaded {len(structured_data)} rows from {file_path}")
        return structured_data
 
    except Exception as e:
        print(f"❌ Error loading Excel file {file_path}: {e}")
        return []
 
# Function to clean text
def clean_text(text):
    text = text.encode("utf-8", "ignore").decode("utf-8")  # Remove invalid characters
    text = re.sub(r"[^\x20-\x7E]", " ", text)  # Remove non-ASCII characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text
 
# Load PowerPoint Files
def load_ppts(ppt_files):
    docs = []
    for file in ppt_files:
        try:
            loader = UnstructuredPowerPointLoader(file)
            slides = loader.load()
            docs.extend([s.page_content for s in slides])
        except Exception as e:
            print(f"❌ Error loading PowerPoint {file}: {e}")
    return docs
 
# Load all documents and store in FAISS
def load_all_data():
    # Load Excel Files
    excel_data1 = load_excel(r"C:\Users\perna\Downloads\Apps\CRONOGRAMA DE ATIVIDADE2.xlsx")
    excel_data2 = load_excel(r"C:\Users\perna\Downloads\conversation_training_data_updated.xlsx")
    excel_data3 = load_excel(r"C:\Users\perna\Downloads\Master DataSet 2.xlsx")
 
    # Combine all documents
    documents = excel_data1 + excel_data2 + excel_data3
 
    if not documents:
        print("⚠️ No valid documents found!")
        return
 
    print(f"📂 Total documents loaded: {len(documents)}")
 
    # Ensure each row is treated as a whole chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50,separators= " |")  # Large chunks to avoid splitting rows
    chunks = splitter.create_documents(documents)
 
    if len(chunks) == 0:
        print("⚠️ No chunks generated! Check document format.")
        return
 
    print(f"📄 Total chunks after processing: {len(chunks)}")
 
    # Convert to embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embedding_model)
 
    # Save FAISS index
    vector_store.save_local("company_faiss_index")
    print("✅ FAISS index successfully saved!")
 
if __name__ == "__main__":
    load_all_data()