import pandas as pd
import pyarrow.parquet as pq
import os
from tqdm import tqdm

def chunk_text(text, chunk_size=256, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.split()) > 50:
            chunks.append(chunk)
    return chunks if chunks else [text]

def process_triplets(triplet_path, output_dir, batch_size=10000):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_parquet(triplet_path)
    
    all_chunks = []
    batch_id = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        for chunk_idx, chunk in enumerate(chunk_text(row['positive'])):
            all_chunks.append({
                'chunk': chunk,
                'document': f"pos_{idx}",
                'row_index': len(all_chunks),
                'chunk_index': chunk_idx,
                'url': f"msmarco:pos_{idx}"
            })
            
        if len(all_chunks) >= batch_size:
            pd.DataFrame(all_chunks[:batch_size]).to_parquet(f"{output_dir}/{batch_id}.parquet")
            all_chunks = all_chunks[batch_size:]
            batch_id += 1
    
    if all_chunks:
        pd.DataFrame(all_chunks).to_parquet(f"{output_dir}/{batch_id}.parquet")

if __name__ == "__main__":
    process_triplets("../data/msmarco_triplets.parquet", "../data/parquet_chunks")