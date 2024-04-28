import os
import pandas as pd
# import dask.dataframe as dd
from utils import aws

def __load_data(bucket_name, s3_file_key, download_path, use_cache=False, return_pandas=False):
    """
    Load the data from S3 into a Dask dataframe (default action).
    
    :param bucket_name: s3 bucket name (required).
    :param s3_file_key: s3 file path (key in AWS language) (required).
    :param download_path: local filepath to store file (required).
    :param use_cache: bool. Use the cached file in data directory.
        If the file is not found, it is downloaded. (Default is False, downloads from AWS)
    :param return_pandas: bool. Return a pandas dataframe instead of a dask dataframe
        (default False, returns dask)
    """
    if use_cache:
        if not os.path.exists(download_path):
            aws.download_s3_file(bucket_name, s3_file_key, download_path)
        lob_dd = dd.read_parquet(download_path)
    else:
        aws.load_s3_file_as_ddf('s3://dsmp-ol2/processed-data/lob_sample_data.parquet')

    if return_pandas:
        return lob_dd.compute()
    else:
        return lob_dd

def load_sample(use_cache=False, return_pandas=False):
     if use_cache:
        if not os.path.exists('../data/lob/lob_sample_data.parquet'):
            aws.download_s3_file('dsmp-ol2', 'processed-data/lob_sample_data.parquet', '../data/lob/lob_sample_data.parquet')
        lob_dd = dd.read_parquet('../data/lob/lob_sample_data.parquet')
    else:
        aws.load_s3_file_as_ddf('s3://dsmp-ol2/processed-data/lob_sample_data.parquet')

    if return_pandas:
        return lob_dd.compute()
    else:
        return lob_dd
    