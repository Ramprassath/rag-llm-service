from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import pickle
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Load embedding model ONCE
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index ONCE
index = faiss.read_index("faiss_index")

# Load metadata
with open("faiss_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

class Query(BaseModel):
    query: str
    k: int = 5

@app.post("/retrieve")
def retrieve(q: Query):
    emb = embedder.encode([q.query]).astype("float32")
    _, ids = index.search(emb, q.k)

    context = "\n\n".join(metadata[i]["text"] for i in ids[0])
    return {"context": context}

@app.get("/health")
def health():
    return {"status": "ok"}
