import os
import uuid
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

# Initialize BERT model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased").to(device)

def generate_embedding(text):
    """Generate BERT embedding for text"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                       padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

def store_embedding(embedding, metadata):
    """Store embedding and metadata in Pinecone"""
    unique_id = str(uuid.uuid4())
    if isinstance(embedding, np.ndarray):
        if embedding.ndim > 1:
            embedding = embedding.flatten()
        embedding_list = embedding.tolist()
    elif isinstance(embedding, list):
        if any(isinstance(i, list) for i in embedding):
            embedding_list = [item for sublist in embedding for item in sublist]
        else:
            embedding_list = embedding
    else:
        raise ValueError("Embedding must be a numpy array or a list")
    
    embedding_list = [float(val) for val in embedding_list]
    index.upsert(vectors=[(unique_id, embedding_list, metadata)])