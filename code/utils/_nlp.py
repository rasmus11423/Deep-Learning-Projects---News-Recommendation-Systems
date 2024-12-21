try:
    from transformers import AutoModel
except ImportError:
    print("transformers not available")

def get_transformers_word_embeddings(model: AutoModel):
    """From EBNerd-Benchmark."""
    return model.embeddings.word_embeddings.weight.data.to("cpu").numpy()
