# embed_docs.py
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os

# Paths
DATA_PATH = "data/healthcare_docs.csv"
INDEX_PATH = "embeddings/healthcare_index.faiss"
DOCS_PATH = "embeddings/docs.pkl"

# Create embeddings folder if not exists
os.makedirs("embeddings", exist_ok=True)

# Load documents
df = pd.read_csv(DATA_PATH)

# Initialize embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings
embeddings = model.encode(df['content'].tolist())

# Save embeddings in FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings, dtype=np.float32))
faiss.write_index(index, INDEX_PATH)

# Save dataframe
with open(DOCS_PATH, "wb") as f:
    pickle.dump(df, f)

print("FAISS index and document pickle saved!")
