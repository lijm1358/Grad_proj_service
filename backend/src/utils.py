import datetime
import os

from google.cloud import storage


def get_current_date_str() -> str:
    now = datetime.datetime.now()
    return now.strftime("%Y%m%d")


def gcp_storage_conn() -> storage.Bucket:
    storage_client = storage.Client()

    bucket_name = os.environ["GCP_BUCKET_NAME"]
    bucket = storage_client.bucket(bucket_name)

    return bucket


def gcp_storage_upload(bucket: storage.Bucket, file_upload: str, dest_path: str):
    source_file_name = file_upload
    destination_blob_name = dest_path

    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
