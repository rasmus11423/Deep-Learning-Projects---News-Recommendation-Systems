from utils._azure import download_datasets
import sys

# Check if the correct number of arguments is provided
if len(sys.argv) != 2:
    print("Usage: python azurefoo.py <datasetname>")
    sys.exit(1)

# Retrieve the dataset name from command line
datasetname = sys.argv[1]

# Use the dataset name in your script
print(f"Dataset name received: {datasetname}")

download_datasets(datasetname)
