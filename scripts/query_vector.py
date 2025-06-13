#/Users/adamkahirov/Desktop/code/re:ranker/scripts/query_vector.py
from typing import List
# import torch
# import sentence_transformers
import requests
import os

class Vectorize:
    def __init__(self, model_path:str):
        self.model_path = model_path
        # self._encoder = sentence_transformers.SentenceTransformer(
        #     model_name_or_path=model_path,
        #     device='cpu',
        #     trust_remote_code=True
        # )
    def encode(self, query:str) -> List[float]:
        #return self._encoder.encode(query).tolist()
        return requests.request(method="POST", 
            url=os.getenv("EMBEDDINGS_ENDPOINT", "http://localhost:8000/v1/embeddings"),
            json={
                "model": self.model_path,
                "input": [ query ]
            }, 
            headers={"Content-Type": "application/json"},
            verify=False).json()["data"][0]["embedding"]