#/Users/adamkahirov/Desktop/code/re:ranker/scripts/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Tuple
import os
import sys
import data_fetch
from query_vector import Vectorize
import asyncio
import multiprocessing
from functools import partial
import concurrent
import time
import numpy as np
import requests
import logging
import uvicorn
from operator import itemgetter

from dotenv import load_dotenv
load_dotenv()
print(f"DEBUG: SEARCH_ENDPOINT is currently set to -> {os.getenv('SEARCH_ENDPOINT')}") # <-- ADD THIS LINE


# Prologue for server set up and env variable handling

# NOTE: much code borrowed from https://github.ibm.com/alanbraz/simple-search-api/blob/main/main.py
app = FastAPI()

logging.basicConfig(level=os.environ.get('UVICORN_LOG_LEVEL', logging.DEBUG))   # add this line
logger = logging.getLogger(__name__)
logger.info("API is starting up")
logger.info(uvicorn.Config.asgi_version)

MODEL_PATH = os.environ.get('MODEL_PATH')
if MODEL_PATH is None:
    MODEL_PATH = 'NO_MODEL_ENV'
try:
    vectorize = Vectorize(MODEL_PATH)
except OSError:
    logger.warning(f"Warning, model {MODEL_PATH} not found")
    vectorize = None

WORKER_POOL_SIZE = int(os.getenv('WORKER_POOL_SIZE', '10'))
logger.info(WORKER_POOL_SIZE)
worker_pool = concurrent.futures.ProcessPoolExecutor(WORKER_POOL_SIZE)
        
DATA_DIR = os.getenv('DATA_DIR', os.getcwd())
# Expect a ',' separated string
# NOTE: for common crawl, this should be url
PARQUET_METADATA = (os.getenv('PARQUET_METADATA', 'document')).split(',')
parquest_set = data_fetch.ParquetSet(DATA_DIR, PARQUET_METADATA)
fetch_parquet_chunk = partial(data_fetch.ParquetSet.get_chunk_and_metadata, parquest_set)

# FastAPI types
class SearchRequest(BaseModel):
    text: str
    limit: Optional[int] = 10

class SearchResult(BaseModel):
    id: int #ChunkID
    content: str # chunk content
    url: str
    score: float
    extra: Optional[dict]
    
class SearchResponse(BaseModel):
    stats: dict
    results: List[SearchResult]

async def run_in_process(func, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(worker_pool, func, *args)

@app.post("/search")
async def search(request: SearchRequest) -> SearchResponse:
    start_time = time.perf_counter()
    # Replace this with your actual search logic
    if vectorize is None:
        raise HTTPException(status_code=404, detail="Vector model not found")
    # string to embeddings vector
    embed_start = time.perf_counter()
    vector = vectorize.encode(request.text)
    embed_time = (time.perf_counter() - embed_start) * 1000  # Convert to milliseconds
    logger.debug("vector", str(vector))
    
    # convert vector to np
    bin_vector_start = time.perf_counter()
    query_vector = np.array(vector, dtype=np.float32)
    binary_data = query_vector.tobytes()
    bin_vector_time = (time.perf_counter() - bin_vector_start) * 1000  # Convert to milliseconds
    logger.debug("query_vector", str(query_vector))
    # logger.debug("binary_data", str(binary_data))
    
    # call search
    search_start = time.perf_counter()
    query = {"query": ("vector.bin", binary_data, "application/octet-stream")}
    response = requests.post(os.getenv("SEARCH_ENDPOINT", "http://localhost:8000/search_vector/"), files=query)
    logger.debug("search_vector", str(response.json()))
    search_time = (time.perf_counter() - search_start) * 1000  # Convert to milliseconds
    response_json = response.json()
    # sort score desc
    sorted_results = sorted(response_json, key=itemgetter(2), reverse=True)
    logger.debug(sorted_results, str(sorted_results))
    # retrieve parquet files
    retrieve_start = time.perf_counter()

    results = []
    tasks = [run_in_process(fetch_parquet_chunk, parquet_batch, row_index) for parquet_batch, row_index, ignore in sorted_results]
    task_results = await asyncio.gather(*tasks)
    for i, r in enumerate(task_results):
    # for i in range(len(sorted_results)):
        # logger.debug(i, str(r))
        # HACK: the reason for this swtich has to do with the different metadata structure the we find in cc for arxiv
        if 'document' in r:
            url = r['document']
        else:
            url = r['url']
        # TODO: fix the id field
        results.append(SearchResult(id=i+1, content=r['chunk'], url=url, score=sorted_results[i][2], extra = {'parquet_batch': sorted_results[i][0], 'row_index': sorted_results[i][1]}))
    retrieve_time = (time.perf_counter() - retrieve_start) * 1000  # Convert to milliseconds
    
    # get titles from arxiv api, if needed
    
    return SearchResponse(stats= { 
                                    "embed_time": embed_time,
                                    "bin_vector_time": bin_vector_time,
                                    "search_time": search_time,
                                    "retrieve_time": retrieve_time,
                                    "total_time": (time.perf_counter()-start_time)*1000,
                                    }, results=results )

@app.get("/stats")
async def stats():
    return { "documents": 1833413 } #, "vectors": 74000000 }
      
class ChunkRequest(BaseModel):
    batch_id: int
    absolute_row_idx: List[int]

@app.post("/get_chunk_test/")
async def get_chunk_test(request:ChunkRequest):
    #num_rows = len(request.absolute_row_idx)
    #aboslute_access_tuples = list(zip([request.batch_id] * num_rows, request.absolute_row_idx))
    #result = parallel_reader.process_absolute_access_tuples(worker_pool, aboslute_access_tuples, use_async=True)
    #loop = asyncio.get_running_loop()
    #result = await loop.run_in_executor(None, result.get)
    #return result

    
    tasks = [run_in_process(fetch_parquet_chunk, request.batch_id, idx) for idx in request.absolute_row_idx]

    results = await asyncio.gather(*tasks)
    return results

    


    

@app.get("/")
async def root():
    return {"message": "Hello World"}