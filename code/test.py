from pathlib import Path
import zipfile

# Define file paths
predictions_dir = Path(__file__).resolve().parent.parent / "data" / "predictions"
predictions_file = predictions_dir / "predictions.txt"
zip_file = predictions_dir / "predictions.zip"

# Zip the predictions.txt file
if predictions_file.exists():
    with zipfile.ZipFile(zip_file, "w") as zipf:
        zipf.write(predictions_file, arcname="predictions.txt")
    print(f"Zipped file created at: {zip_file}")
else:
    print(f"File not found: {predictions_file}")

# Unzip and display contents
if zip_file.exists():
    with zipfile.ZipFile(zip_file, "r") as zipf:
        zipf.extractall(predictions_dir)
        print("Unzipped the file successfully!")

    # Read and display predictions.txt
    with open(predictions_file, "r") as f:
        print("---- Predictions File Contents ----")
        print(f.read())
else:
    print("Zip file not found!")
