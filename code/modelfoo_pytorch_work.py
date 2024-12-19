import argparse
import neptune
import os
from pathlib import Path
import torch
import torch.nn as nn
from models.dataloader_pytorch import NRMSDataLoader

from tqdm import tqdm
from torch.utils.data import DataLoader
from models.data_helper_work import grab_data,grab_embeded_articles,initialize_model,train_model,validate_model
from utils._constants import (
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_IMPRESSION_ID_COL,
)
from utils._python import write_submission_file, rank_predictions_by_score

torch.cuda.empty_cache()

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train a model with optional Neptune logging.")
parser.add_argument("--debug", action="store_true", help="Run the script in debug mode (no Neptune logging).")
parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to use for training.")
args = parser.parse_args()

# Access the dataset name
dataset_name = args.dataset
print(f"Using dataset: {dataset_name}")

# Initialize Neptune run unless in debug mode
if not args.debug:
    # Get API token from environment variable
    api_token = os.getenv("NEPTUNE_API_TOKEN")
    if api_token is None:
        raise ValueError("Neptune API token not found. Please set the 'NEPTUNE_API_TOKEN' environment variable.")


    run = neptune.init_run(
        project="Deep-learning-project/model-training",
        api_token=api_token,
        )

    print("neptune intialized")
else:
    run = ""
    print("Debug mode: Neptune logging is disabled.")

# set parameters 
HISTORY_SIZE = 20
BATCH_SIZE = 32
title_size, history_size = 30, 30
head_num, head_dim, attention_hidden_dim, dropout = 20, 20, 200, 0.2

BASE_PATH = Path(__file__).resolve().parent.parent
LOCAL_TOKENIZER_PATH = BASE_PATH.joinpath("data/local-tokenizer")
LOCAL_MODEL_PATH = BASE_PATH.joinpath("data/local-tokenizer-model")


## Grab Data
print("Grabbing train and validation set.")

df_train, df_validation = grab_data(dataset_name,HISTORY_SIZE)


print("Grabbing articles and embeddings")
article_mapping, word2vec_embedding = grab_embeded_articles(LOCAL_TOKENIZER_PATH,LOCAL_MODEL_PATH,dataset_name, title_size)


# Initialize Dataloaders
train_dataloader = NRMSDataLoader(
    behaviors=df_train,
    article_dict=article_mapping,
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    unknown_representation="zeros",
    eval_mode=False,
    batch_size=BATCH_SIZE, 
)

val_dataloader = NRMSDataLoader(
    behaviors=df_validation,
    article_dict=article_mapping,
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    unknown_representation="zeros",
    eval_mode=True,
    batch_size=BATCH_SIZE,
)

# Wrap in PyTorch DataLoader
train_loader = DataLoader(train_dataloader, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataloader, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

model = initialize_model(word2vec_embedding, title_size, HISTORY_SIZE, head_num, head_dim, attention_hidden_dim, dropout)

print(f"Loaded word2vec embedding shape: {word2vec_embedding.shape}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Selected device: {device}")
model.to(device)

# Set up optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) # with added weight decay
criterion = nn.CrossEntropyLoss()

if not args.debug:
    params = {
        "optimizer": "Adam",
        "learning_rate":0.001,
        "dataset": dataset_name,
        "batchsize": BATCH_SIZE
        }
    run["parameters"] = params

# Training and validation loop
epochs = 1
for epoch in range(epochs):
    # Train the model
    with tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}") as pbar:
            train_loss, train_acc, train_auc = train_model(pbar, model, criterion, optimizer, device, args, run)
    print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}, Train Auc = {train_auc:.4f}")

    # Validate the model
    with tqdm(val_dataloader, desc=f"Validation Epoch {epoch + 1}") as pbar:
            val_loss, val_acc, val_auc, pred_validation = validate_model(pbar, model, criterion, device,args, run)

    print(f"Epoch {epoch + 1}: Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}, Val Auc = {val_auc:.4f}")

import polars as pl

from utils._behaviors import (
    add_prediction_scores,
    add_known_user_column
)

from utils._constants import (
    DEFAULT_USER_COL,
)




# Add prediction scores and mark known users
df_validation = add_prediction_scores(df_validation, pred_validation).pipe(
    add_known_user_column, known_users=df_train[DEFAULT_USER_COL]
)

df_validation.head(2)

# Rank predictions and write submission file
print("Ranking predictions and writing submission file...")
df_validation = df_validation.with_columns(
    pl.col("scores")
    .map_elements(lambda x: list(rank_predictions_by_score(x)))
    .alias("ranked_scores")
)

write_submission_file(
    impression_ids=df_validation[DEFAULT_IMPRESSION_ID_COL],
    prediction_scores=df_validation["ranked_scores"],
    path="downloads/predictions.txt",
)

print("Submission file successfully created.")


if not args.debug:
    run.stop()

print("Done")
torch.cuda.empty_cache()