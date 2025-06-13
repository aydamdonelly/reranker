from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List
import os

app = FastAPI()

model = SentenceTransformer(os.getenv('MODEL_PATH', 'ibm-granite/granite-embedding-30m-english'), device='cuda')

class EmbeddingRequest(BaseModel):
    model: str
    input: List[str]

@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    embeddings = model.encode(request.input)
    return {
        "data": [{"embedding": emb.tolist(), "index": i} for i, emb in enumerate(embeddings)],
        "model": request.model,
        "object": "list"
    }

@app.get("/")
async def root():
    return {
        "status": "running",
        "model": model.get_config()['model_name_or_path'],
        "device": str(model.device),
        "dimension": model.get_sentence_embedding_dimension()
    }