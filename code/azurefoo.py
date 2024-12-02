import os, uuid
from pathlib import Path
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

def requireDatasets():
    BASE_PATH = Path(__file__).resolve().parent
    DATA_PATH = BASE_PATH.joinpath("data")

    if not Path.is_dir(DATA_PATH.join("eberd_demo")):
        download_datasets()
        print("Datasets downloaded")
    else: print("Datasets already downloaded")
    

def download_datasets():
    '''Function which downloads dataset from azure'''

    BASE_PATH = Path(__file__).resolve().parent
    DATA_PATH = BASE_PATH.joinpath("data")

    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        raise ValueError("Azure Storage connection string not found in environment variables. Should be AZURE_STORAGE_CONNECTION_STRING")

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_name = "ebrec-demo"

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
