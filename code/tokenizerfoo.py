from transformers import AutoTokenizer, AutoModel
import os
from pathlib import Path

# Define local paths
BASE_PATH = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_PATH.joinpath("data")
LOCAL_TOKENIZER_PATH = DATA_PATH.joinpath("local-tokenizer")
LOCAL_MODEL_PATH = DATA_PATH.joinpath("local-tokenizer-model")

print("ackanowledge me damnit")

# Step 1: Download and save the tokenizer and model locally (only required once)
def download_and_save_transformer():
    print("Downloading tokenizer and model from Hugging Face...")
    
    # Download tokenizer and save locally
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
    tokenizer.save_pretrained(LOCAL_TOKENIZER_PATH)
    
    # Download model and save locally
    model = AutoModel.from_pretrained("FacebookAI/xlm-roberta-base")
    model.save_pretrained(LOCAL_MODEL_PATH)
    
    print(f"Tokenizer saved to {LOCAL_TOKENIZER_PATH}")
    print(f"Model saved to {LOCAL_MODEL_PATH}")

# Step 2: Load tokenizer and model from local directory
def load_transformer_from_local():
    print("Loading tokenizer and model from local directories...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_TOKENIZER_PATH)
    
    # Load model
    model = AutoModel.from_pretrained(LOCAL_MODEL_PATH)
    
    print("Tokenizer and model loaded successfully.")
    return tokenizer, model

# Step 3: Use the tokenizer to tokenize input text
def tokenize_text(tokenizer, text):
    print("Tokenizing text...")
    tokenized_output = tokenizer(
        text,
        max_length=30,
        truncation=True,
        padding="max_length",
        return_tensors="pt"  # Return PyTorch tensors
    )
    print("Tokenized output:", tokenized_output)
    return tokenized_output

if __name__ == "__main__":
    # Check if tokenizer and model are already saved locally
    if not (os.path.exists(LOCAL_TOKENIZER_PATH) and os.path.exists(LOCAL_MODEL_PATH)):
        download_and_save_transformer()
    
    # Load tokenizer and model from local
    tokenizer, model = load_transformer_from_local()
    
    # Example text to tokenize
    text = "This is an example sentence for the tokenizer to process."
    
    # Tokenize the text
    tokenized_output = tokenize_text(tokenizer, text)
