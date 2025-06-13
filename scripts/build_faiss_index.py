import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
import pandas as pd
import glob
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class FAISSIndexBuilder:
    def __init__(self, model_name='ibm-granite/granite-embedding-30m-english'):
        self.model = SentenceTransformer(model_name, device='cuda')
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.model_name = model_name
        
    def build_index(self, parquet_dir, index_dir):
        os.makedirs(index_dir, exist_ok=True)

        base_index = faiss.IndexFlatL2(self.dimension)
        cpu_index = faiss.IndexIDMap(base_index)
        
        id_mapping = []
        
        for parquet_file in tqdm(sorted(glob.glob(f"{parquet_dir}/*.parquet"))):
            batch_id = int(os.path.basename(parquet_file).split('.')[0])
            df = pd.read_parquet(parquet_file)
            
            embeddings = self.model.encode(df['chunk'].tolist(), batch_size=256, show_progress_bar=True).astype('float32')
            
            start_id = len(id_mapping)

            cpu_index.add_with_ids(embeddings, np.arange(start_id, start_id + len(embeddings)))
            
            id_mapping.extend([(batch_id, i) for i in range(len(embeddings))])
        
        faiss.write_index(cpu_index, f"{index_dir}/chunks.index")
        
        with open(f"{index_dir}/id_mapping.pkl", "wb") as f:
            pickle.dump(id_mapping, f)
        
        with open(f"{index_dir}/metadata.pkl", "wb") as f:
            pickle.dump({
                'model': self.model_name, 
                'dimension': self.dimension,
                'total_vectors': len(id_mapping),
                'index_type': 'IndexIDMap(IndexFlatL2)' # Geändert für Klarheit
            }, f)

if __name__ == "__main__":
    FAISSIndexBuilder().build_index("../data/parquet_chunks", "../data/faiss_index")