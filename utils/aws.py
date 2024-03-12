import boto3
import dask.dataframe as dd

# Instantiate session from ~/.aws/credentials for S3 access
session = boto3.Session(profile_name="default")


def load_s3_file_as_ddf(file_name):
    ddf = dd.read_parquet(file_name, storage_options={"profile": "default"})
    return ddf
