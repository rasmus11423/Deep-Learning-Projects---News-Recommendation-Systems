# **Neural News Recommendation System**  
**Deep Learning Project**  
**Group 14**: s181486, s215160, s215231  
**Course**: 02456 (Fall 2024)

This project implements a **Neural News Recommendation System (NRMS)** using the **Ekstra Bladet News Recommendation Dataset (EB-NeRD)**. The system predicts which articles a user is most likely to click on by encoding both user behavior and article content with **multi-headed self-attention** mechanisms.

---

## **Overview**

- **Dataset**: Ekstra Bladet News Recommendation Dataset (EB-NeRD)  
  - Contains user behavior logs, article metadata, and extended features (e.g., user demographics, named entities, sentiment).

- **Model Architecture**: Based on the NRMS approach by Wu et al. [1], featuring:
  - **News Encoder**: Transforms each article’s title into a vector representation using:
    - Embedding layer  
    - Multi-head self-attention  
    - Additive self-attention
  - **User Encoder**: Creates a user embedding by aggregating the representations of previously read articles via multi-head self-attention.
  - **Click Predictor**: Computes similarity scores between the user vector and candidate article vectors, selecting the article with the highest score.

- **Implementation**:
  - Written in **PyTorch**
  - Converted from a TensorFlow-based reference implementation
  - Hyperparameter tuning (e.g., learning rate, weight decay, number of attention heads) to reduce overfitting and improve model accuracy

- **Performance**:
  - Best AUC ~0.568 on the validation set (with limited training data due to memory constraints)
  - Negative sampling and cross-entropy loss improved the model’s training performance

---

## **Key Files**

### `dummyModel.ipynb`
- Initializes and trains the NRMS model with minimal data  
- Illustrates the PyTorch-based workflow for data loading, tokenizing, and running the training loop

### `tokenizerDownloader.py`
- Downloads the **XLM-RoBERTa-base** tokenizer from Hugging Face  
- Ensures that the correct tokenizer files exist in the `data/` folder for subsequent model training

### `ebnerd-demo` (within `data/`)
- A small sample dataset for demo or testing  
- Full dataset(s) are typically larger and must be placed here (or adapt the notebook if you store them elsewhere)

---

## **Model Details**

- **Negative Sampling & Cross-Entropy Loss**:
  - Implemented to address classification of clicked vs. unclicked articles
  - Helps reduce overfitting by comparing clicked items to randomly sampled unclicked ones

- **Attention Mechanisms**:
  - **Multi-head self-attention** captures contextual relationships between words (in articles) and between articles (in user history)
  - Incorporates an **additive attention** layer to emphasize the most relevant words or articles in the final encoding

---

## **Limitations & Future Work**

- **Memory Constraints**:
  - The PyTorch DataLoader for EB-NeRD can be memory-intensive
  - We used a smaller sample for quick training, limiting maximum achievable performance

- **Extending the User Encoder**:
  - Additional features (user demographics, sentiment analysis, etc.) could improve personalization

- **Regularization & Early Stopping**:
  - Overfitting often appeared after ~15 epochs
  - Early stopping or more aggressive regularization could further stabilize training

---

## **References**

[1] **Mingxiao An, Fangzhao Wu, Chuhan Wu, Kun Zhang, Zheng Liu, and Xing Xie.**  
“Neural news recommendation with long- and short-term user representations.”  
*Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, 2019.

---

Enjoy exploring the **Neural News Recommendation System**! If you have any questions or discover improvements, feel free to open an issue or submit a pull request.
