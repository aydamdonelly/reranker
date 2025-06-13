    from fastapi import FastAPI, File, UploadFile
    import faiss
    import numpy as np
    import pickle

    app = FastAPI()

    cpu_index = faiss.read_index("../data/faiss_index/chunks.index")
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

    with open("../data/faiss_index/id_mapping.pkl", "rb") as f:
        id_mapping = pickle.load(f)

    with open("../data/faiss_index/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    @app.post("/search_vector/")
    async def search_vector(query: UploadFile = File(...)):
        query_vector = np.frombuffer(await query.read(), dtype=np.float32).reshape(1, -1)
        distances, indices = index.search(query_vector, 10)
        
        return [[id_mapping[idx][0], id_mapping[idx][1], float(1/(1+dist))] 
                for idx, dist in zip(indices[0], distances[0]) if idx != -1]

    @app.get("/")
    async def root():
        return {
            "status": "running",
            "index_loaded_on_gpu": True,
            "total_vectors": metadata['total_vectors']
        }