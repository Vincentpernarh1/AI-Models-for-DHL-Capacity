import os
import time
import threading
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA

# Flask app
app = Flask(__name__)
CORS(app, resources={r"/ask": {"origins": "*"}, r"/healthcheck": {"origins": "*"}})

# Globals
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = None
llm = None
qa = None
retriever = None

# Paths
vector_store_path = os.getenv("VECTOR_STORE_PATH", "company_faiss_index")
llm_model_path = os.getenv("LLM_MODEL_PATH", "C:/Users/perna/Desktop/AI/mistral-7b-instruct-v0.2.Q4_K_M.gguf")

# Load FAISS vector store
def load_vector_store():
    global vector_store
    try:
        print("🔄 Loading FAISS vector store...")
        if os.path.exists(vector_store_path):
            vector_store = FAISS.load_local(vector_store_path, embedding_model, allow_dangerous_deserialization=True)
            print(f"✅ FAISS vector store loaded from {vector_store_path}")
        else:
            raise FileNotFoundError(f"❌ FAISS index not found at {vector_store_path}")
    except Exception as e:
        print(f"Error loading FAISS index: {e}")

# Load Llama model
def load_llm():
    global llm
    try:
        print("🔄 Loading Llama model...")
        if os.path.exists(llm_model_path):
            llm = LlamaCpp(
                model_path=llm_model_path,
                system_prompt="Answer ONLY based on the data. Give only direct answers. DO NOT provide explanations, context, or extra details. If no direct answer exists, say 'No relevant data found.'",
                temperature=0.2,
                # n_gpu_layers=-1, this will use all available GPU memory
                n_ctx=1024,
                top_p=0.95,
                max_tokens=200,
                n_batch=512,
                verbose=False,
            )
            print("✅ LlamaCpp model loaded successfully.")
        else:
            raise FileNotFoundError(f"❌ Model file not found: {llm_model_path}")
    except Exception as e:
        print(f"Error loading LlamaCpp model: {e}")

# Initialize QA chain
def initialize_qa_chain():
    global qa, retriever
    try:
        print("🔄 Initializing QA chain...")
        if vector_store and llm:
            retriever = vector_store.as_retriever(search_type="similarity" , search_kwargs={"k": 2})
            qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
            print("✅ QA chain initialized and ready.")
        else:
            raise Exception("Models not loaded.")
    except Exception as e:
        print(f"Error initializing QA chain: {e}")

# Background model loading
def init_models_background():
    print("🚀 Starting background model initialization...")
    load_vector_store()
    load_llm()
    initialize_qa_chain()
    print("✅ All models initialized.")

# Run model loading in background
threading.Thread(target=init_models_background).start()

@app.route("/healthcheck")
def healthcheck():
    if all([llm, vector_store, qa]):
        return jsonify({"status": "ready"}), 200
    return jsonify({"status": "initializing"}), 503

@app.route("/ask", methods=["POST"])
def ask_question():
    try:
        data = request.get_json()
        question = data.get("question", "").strip().lower()

        if not question:
            return jsonify({"error": "No question provided"}), 400
        if not all([llm, vector_store, qa]):
            return jsonify({"error": "QA system still initializing"}), 503

        def generate_response():
            try:
                # 1. Immediately send a THINKING signal
                yield "data: [THINKING]\n\n"
                
                # 2. Perform the slow query processing
                print(f"🔍 Processing question: {question}")
                response = qa.invoke(question)
                answer = response.get("result", "").strip()

                # 3. Stream the actual answer
                if not answer or "no relevant data found" in answer.lower():
                    yield "data: No relevant data found.\n\n"
                else:
                    for word in answer.split():
                        yield f"data: {word} \n\n"
                        time.sleep(0.02)
                
                # 4. Signal the end of the stream
                yield "data: [END]\n\n"
            except Exception as e:
                print(f"⚠️ Streaming error: {e}")
                yield f"data: ⚠️ Error: {str(e)}\n\n"
                yield "data: [END]\n\n"

        return Response(generate_response(), content_type="text/event-stream")

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)