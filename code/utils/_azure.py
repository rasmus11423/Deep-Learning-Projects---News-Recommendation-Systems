import os, uuid
from pathlib import Path
from azure.storage.blob import BlobServiceClient

def download_datasets(dataset):
    '''Function which downloads dataset from azure'''
    datasets_choices = ["ebrec-demo","ebnerd-small","ebnerd-test"]

    if dataset not in datasets_choices:
        raise ValueError(f"Dataset not in cloud: choose from: {datasets_choices}")
        
    BASE_PATH = Path(__file__).resolve().parent.parent.parent
    DATA_PATH = BASE_PATH.joinpath("data")

    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        raise ValueError("Azure Storage connection string not found in environment variables. Should be AZURE_STORAGE_CONNECTION_STRING")

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_name = dataset

    container_client = blob_service_client.get_container_client(container_name)

    blob_list = container_client.list_blobs()
    for blob in blob_list:
        print("\t" + blob.name)
    
        download_file_path = DATA_PATH.joinpath(blob.name)
        os.makedirs(download_file_path.parent, exist_ok=True)

        # Download the blob
        blob_client = container_client.get_blob_client(blob.name)
        with open(download_file_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())

def download_tokenizer():
    BASE_PATH = Path(__file__).resolve().parent.parent.parent
    DATA_PATH = BASE_PATH.joinpath("data")

    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        raise ValueError("Azure Storage connection string not found in environment variables. Should be AZURE_STORAGE_CONNECTION_STRING")

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_name = "tokenizer"

    container_client = blob_service_client.get_container_client(container_name)

    blob_list = container_client.list_blobs()
    for blob in blob_list:
        
    
        download_file_path = DATA_PATH.joinpath(blob.name)
        if "model" in blob.name:
            print("\t" + blob.name)
            os.makedirs(download_file_path.parent, exist_ok=True)

            # Download the blob
            blob_client = container_client.get_blob_client(blob.name)
            with open(download_file_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
