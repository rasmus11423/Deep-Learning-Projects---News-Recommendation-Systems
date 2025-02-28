Neural News Recommendation System
Deep Learning Project
Group 14: s181486, s215160, s215231
Course: 02456 (Fall 2024)

This project implements a Neural News Recommendation System (NRMS) using the Ekstra Bladet News Recommendation Dataset (EB-NeRD). The system predicts which articles a user is most likely to click on by encoding both user behavior and article content with multi-headed self-attention mechanisms.

Overview
- Dataset: Ekstra Bladet News Recommendation Dataset (EB-NeRD)
  - Contains user behavior logs, article metadata, and extended features (e.g., user demographics, named entities, sentiment).
- Model Architecture: Based on the NRMS approach by Wu et al. [1], featuring:
  - News Encoder: Transforms each article’s title into a vector representation using:
    - Embedding layer
    - Multi-head self-attention
    - Additive self-attention
  - User Encoder: Creates a user embedding by aggregating the representations of previously read articles via multi-head self-attention.
  - Click Predictor: Computes similarity scores between the user vector and candidate article vectors, selecting the article with the highest score.
- Implementation:
  - Written in PyTorch
  - Conversions from TensorFlow-based reference implementations
  - Hyperparameter tuning (e.g., learning rate, weight decay, number of attention heads) to reduce overfitting and improve model accuracy
- Performance:
  - Best AUC ~0.568 on the validation set (with limited training data due to memory constraints)
  - Negative sampling and cross-entropy loss improved the model’s training performance

Repository Structure
.
├── data/
│   ├── ebnerd-demo/            # Example dataset (not included in this repo for size/privacy)
│   ├── tokenizerDownloader.py   # Script to download XLM-RoBERTa-base tokenizer
│   └── ...                     # Place your dataset and tokenizer files here
├── dummyModel.ipynb            # Minimal example notebook to initialize & train NRMS
├── README.md                   # This file
└── requirements.txt            # Project dependencies

Key Files
dummyModel.ipynb
- Initializes and trains the NRMS model with minimal data
- Illustrates the PyTorch-based workflow for data loading, tokenizing, and running the training loop

tokenizerDownloader.py
- Downloads the XLM-RoBERTa-base tokenizer from Hugging Face
- Ensures that the correct tokenizer files exist in the data/ folder for subsequent model training

ebnerd-demo (within data/)
- A small sample dataset for demo or testing
- Full dataset(s) are typically larger and must be placed here (or adapt the notebook if you store them elsewhere)

Getting Started
1. Clone the Repository
git clone <your-repo-url>
cd <repo-folder>

2. Install Dependencies
- Create a new virtual environment (optional but recommended).
- Install required packages:
pip install -r requirements.txt

Alternatively, you can manually install core packages, such as torch, transformers, and any other needed libraries.

3. Download or Place Your Data
- Put the EB-NeRD dataset (or demo subset) in the data/ebnerd-demo directory.
- Run tokenizerDownloader.py to fetch the XLM-RoBERTa-base tokenizer files, if not already present:
python data/tokenizerDownloader.py

4. Run the Notebook
- Open dummyModel.ipynb in your preferred environment (Jupyter, VSCode, etc.) and follow the cells to train the NRMS model:
jupyter notebook dummyModel.ipynb

This notebook walks you through data loading, model definition, training, and evaluation.

Model Details
- Negative Sampling & Cross-Entropy Loss:
  - Implemented to deal with classification of clicked vs. unclicked articles
  - Helps reduce overfitting by comparing clicked items to randomly sampled unclicked ones
- Attention Mechanisms:
  - Multi-head self-attention used to capture contextual relationships between words (in articles) and between articles (in user history)
  - Incorporates an additive attention layer to weigh the most relevant words or articles more heavily in the final encoding

Limitations & Future Work
- Memory Constraints:
  - The PyTorch DataLoader for EB-NeRD can be memory-intensive
  - We used a smaller sample to train quickly, limiting maximum achievable performance
- Extending the User Encoder:
  - Additional features (user demographics, sentiment analysis, etc.) could improve personalization
- Regularization & Early Stopping:
  - Overfitting often appeared after ~15 epochs
  - Early stopping or more aggressive regularization could further stabilize training

References
[1] Mingxiao An, Fangzhao Wu, Chuhan Wu, Kun Zhang, Zheng Liu, and Xing Xie.
“Neural news recommendation with long- and short-term user representations.”
Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 2019.

Enjoy exploring the Neural News Recommendation System! If you have any questions or discover improvements, feel free to open an issue or submit a pull request.
