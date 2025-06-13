#/Users/adamkahirov/Desktop/code/re:ranker/scripts/data_fetch.py
from typing import Dict, Iterable, List, Optional, Tuple
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pandas as pd
import os
import sys
import glob
import re
import multiprocessing as mp
from functools import partial


AcessTuple = Tuple[int,int,int] # batch, row, chunk
AbosluteAccessTuple = Tuple[int,int ] # batch, absolute_row
class ParallelChunkReader:
    def __init__(self,
                 ps:'ParquetSet'):
        self._ps = ps

    def _get_url_and_chunk(self, tuple:AbosluteAccessTuple) -> Tuple[str,str]:
        """
        Parallel access by absolute row
        """
        return  url_and_chunk(self._ps.get_row_df(*tuple))

    # NOTE: you can use an executor pool in asyncio instead
    def process_absolute_access_tuples(self, pool: mp.Pool, tuples:List[AbosluteAccessTuple], use_async:bool=False) -> Iterable[Tuple[str,str]]:
        func = partial(ParallelChunkReader._get_url_and_chunk, self)
        result =  pool.map_async(func, tuples)
        if use_async:
            return result
        else:
            return result.get()



class ParquetSet:
    """
    The use of storing keys at object creation time was for sanity checking but comes with a limitation:
        - if you dynmically ingest new data, the system won't pick up on it
        - The way to fix this would be to move out the dictionary initialization and periodically rebuild it
          (or on demand) at set intervals.
          This could be implemented with a ttl (time to live) or an explicit refresh/POST request
    Users can specify `metadata_columns` as an optimization so that fetching a chunk excludes columns
    """
    def __init__(self,
                 data_dir:str,
                 metadata_columns: Optional[List[str]] = None):
        if not os.path.isdir(data_dir):
            raise ValueError(f"{data_dir} not a directory")

        if metadata_columns is None:
            metadata_columns = []
        self._metadata_columns = metadata_columns
        self._fetch_columns = list(set(['chunk', 'row_index', 'chunk_index'] + metadata_columns))
        self._chunk_and_metdata_columns = ['chunk'] + metadata_columns


        self._batch_id_to_file = {} # int -> str
        parquest_re = re.compile(r'(\d+)\.parquet')
        for file_name in glob.glob(os.path.join(data_dir, '*.parquet')):
            base = os.path.basename(file_name)
            match = parquest_re.match(base)
            if match:
                self._batch_id_to_file[int(match.group(1))] = file_name

    def get_pq_table(self, batch_index:int):
        """NOTE: this grabs the whole table, and is not suited for real-time query
        """
        filename = self._batch_id_to_file[batch_index] # NOTE: can fail if batch not there
        dataset = ds.dataset(filename, format='parquet')
        return dataset.to_table()

    def get_chunk_df(self,
                     batch_index:int,
                     row_index:int,
                     chunk_index:int) -> pd.DataFrame:
        """
        Get a chunk frame within the set
        Exceptions: can raise KeyError if batch_index is not found
        NOTE: dataframe can be empty if row_index and chunk_index are out of bounds
        """
        filename = self._batch_id_to_file[batch_index] # NOTE: can fail if batch not there
        return parquet_get_fields(filename, row_index, chunk_index, self._fetch_columns)


    def get_row_df(self,
                   batch_index:int,
                   absolute_row_index:int) -> pd.DataFrame:
        """
        If someone can give us the absolute row index, then we can fetch that for them
        Warning: only contains chunk and url
        """
        filename = self._batch_id_to_file[batch_index] # NOTE: can fail if batch not there
        pq_file = pq.ParquetFile(filename)

        current_row_count = 0
        group_offset = -1
        for i in range(pq_file.num_row_groups):
            rg_md = pq_file.metadata.row_group(i)
            n_row_in_group = rg_md.num_rows
            if current_row_count + n_row_in_group > absolute_row_index:
                # This means we have found the row group that contains the absolute offset
                group_offset = absolute_row_index - current_row_count
                break
            current_row_count += n_row_in_group
        # Only read the required row group

        table = pq_file.read_row_group(i, columns=self._chunk_and_metdata_columns)
        row = table.slice(group_offset, 1)
        return row.to_pandas()

    def get_chunk_and_metadata(self,
                               batch_index:int,
                               absolute_row_index: int) -> Dict:
        """
        Return a a dict with keys `chunk` and others for the medata
        """
        df = self.get_row_df(batch_index, absolute_row_index)
        return df.iloc[0].to_dict()



    def get_batch_dict(self) -> dict:
        return dict(self._batch_id_to_file)



def parquet_get_fields(filename: str,
                       row_index: int,
                       chunk_index: int,
                       metadata_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Efficient parquet querying using PyArrow's filtering before converting to Pandas.
    """
    dataset = ds.dataset(filename, format='parquet')

    # Filter rows before conversion to Pandas
    filtered_table = dataset.to_table(
        #columns=['chunk', 'row_index', 'id', 'chunk_index', 'url'],
        columns=metadata_columns,
        filter=(ds.field('row_index') == row_index) & (ds.field('chunk_index') == chunk_index)
    )
    #chunk, url = df.column('chunk')[0], df.column('url')
    return filtered_table.to_pandas()


def url_and_chunk(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    url = None
    chunk = None
    slice = df[['chunk', 'url']]
    if len(slice) >= 1:
        chunk, url = slice.iloc[0]
    return chunk, url