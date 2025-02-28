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
