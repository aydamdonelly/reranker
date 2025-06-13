from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import glob
import pandas as pd
from tqdm import tqdm
import lancedb

app = FastAPI()

DATA_DIR = os.getenv("DATA_DIR", "../data/parquet_chunks")
DB_URI = os.getenv("BM25_DB_URI", "../data/bm25_lancedb")
TABLE_NAME = os.getenv("BM25_TABLE_NAME", "chunks")
TOP_K_DEFAULT = int(os.getenv("BM25_TOP_K", "10"))

db = lancedb.connect(DB_URI)

if TABLE_NAME in db.table_names():
    table = db.open_table(TABLE_NAME)
else:
    rows: List[pd.DataFrame] = []
    parquet_files = sorted(
        glob.glob(os.path.join(DATA_DIR, "*.parquet")),
        key=lambda p: int(os.path.basename(p).split(".")[0]),
    )
    if not parquet_files:
        raise RuntimeError(f"No parquet files found in {DATA_DIR}")

    for parquet_path in tqdm(parquet_files, desc="Reading parquet chunks"):
        batch_id = int(os.path.basename(parquet_path).split(".")[0])
        df = pd.read_parquet(parquet_path, columns=["chunk", "row_index", "url"])
        df = df.rename(columns={"chunk": "content"})
        df.insert(0, "id", range(len(df)))
        df.insert(1, "batch_id", batch_id)
        rows.append(df)

    all_chunks = pd.concat(rows, ignore_index=True)
    table = db.create_table(TABLE_NAME, data=all_chunks)
    table.create_fts_index("content", use_tantivy=True)

class SearchRequest(BaseModel):
    text: str
    limit: Optional[int] = TOP_K_DEFAULT

class SearchResult(BaseModel):
    id: int
    content: str
    url: Optional[str]
    score: float
    extra: Optional[Dict[str, Any]] = None

class SearchResponse(BaseModel):
    results: List[SearchResult]

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    records = (
        table.search(request.text)
        .limit(request.limit)
        .select(["id", "content", "url", "batch_id", "row_index", "_score"])
        .to_list()
    )
    results: List[SearchResult] = [
        SearchResult(
            id=int(r["id"]),
            content=r["content"],
            url=r.get("url", ""),
            score=float(r["_score"]),
            extra={"batch_id": int(r["batch_id"]), "row_index": int(r["row_index"])}
        )
        for r in records
    ]
    return SearchResponse(results=results)

@app.get("/")
async def root():
    return {
        "status": "running",
        "index_type": "bm25_tantivy",
        "documents": table.count_rows(),
        "db_uri": DB_URI,
    } 