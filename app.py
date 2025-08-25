# app.py
from flask import Flask, request, jsonify
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
import os

# ---------------- CONFIG ---------------- #
openai.api_key = "YOUR_OPENAI_API_KEY"  # Replace with your key
INDEX_PATH = "embeddings/healthcare_index.faiss"
DOCS_PATH = "embeddings/docs.pkl"

# Load FAISS index and docs
index = faiss.read_index(INDEX_PATH)
with open(DOCS_PATH, "rb") as f:
    df = pickle.load(f)

# Initialize embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- FUNCTIONS ---------------- #
def retrieve_docs(query, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), top_k)
    results = df.iloc[indices[0]]['content'].tolist()
    return results

def generate_answer(query):
    context_docs = retrieve_docs(query)
    context_text = "\n".join(context_docs)
    
    prompt = f"Answer the healthcare question using the following context:\n{context_text}\nQuestion: {query}\nAnswer:"
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message['content']

# ---------------- FLASK APP ---------------- #
app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    query = data.get("query")
    if not query:
        return jsonify({"error": "No query provided"}), 400
    answer = generate_answer(query)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(port=5000)
