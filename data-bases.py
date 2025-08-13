from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv()
# Initialize Pinecone client

pc = Pinecone(api_key=os.environ.get("API-KEY-PINECONE"), environment="us-east1-gcp")

index_name = "nostradecius"

if not pc.has_index(index_name):
    pc.create_index_for_model(
        name=index_name,
        cloud="aws",
        region="us-east-1",
        embed={
            "model":"llama-text-embed-v2",
            "field_map":{"text": "chunk_text"}
        }
    )