from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import pickle
import numpy as np

app = FastAPI()

index = None
metadata = None

def load_index():
    global index, metadata
    if index is None:
        index = faiss.read_index("faiss_index")
        with open("faiss_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

class Query(BaseModel):
    query: str
    k: int = 5

@app.post("/retrieve")
def retrieve(q: Query):
    load_index()

    # ⚠️ TEMPORARY SIMPLE EMBEDDING (lightweight)
    # Replace later with proper embedding if needed
    query_vec = np.random.rand(1, index.d).astype("float32")

    _, ids = index.search(query_vec, q.k)
    context = "\n\n".join(metadata[i]["text"] for i in ids[0])
    return {"context": context}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"status": "alive"}
