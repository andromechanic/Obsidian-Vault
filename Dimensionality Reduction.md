---
tags:
  - ML
Date: 16-02-2025 13:19
---

## Dimensionality Reduction

---

## **What is Dimensionality Reduction?**

Dimensionality reduction is the process of reducing the number of features (variables) in a dataset while preserving as much important information as possible. It helps in improving machine learning models by removing redundant or irrelevant features.

---

## **Why Do We Need Dimensionality Reduction?**

1. **Curse of Dimensionality** – When there are too many features, models become slower, require more data, and may overfit.
2. **Visualization** – High-dimensional data (e.g., 100 features) is hard to visualize. Dimensionality reduction can reduce it to 2D or 3D.
3. **Noise Reduction** – Eliminates irrelevant or redundant features, improving model accuracy.
4. **Storage and Computation** – Fewer dimensions mean lower memory and computation requirements.

---

## **Types of Dimensionality Reduction Techniques**

Dimensionality reduction techniques are divided into:

1. **Feature Selection** – Selects a subset of the original features.
2. **Feature Extraction** – Creates new features by transforming the original ones.

Let’s explore the two types in detail.

---

### **1. Feature Selection**

This method selects important features from the original dataset without modifying them.

#### **Methods for Feature Selection**

- **Filter Methods** – Use statistical techniques (e.g., correlation, mutual information) to select important features.
- **Wrapper Methods** – Use machine learning models to evaluate feature importance.
- **Embedded Methods** – Feature selection is done during model training (e.g., Lasso regression).

Example: If you have 100 features in a dataset, feature selection might keep only the 10 most important ones.

---

### **2. Feature Extraction**

This method transforms the original features into a new set of fewer features while retaining most of the important information.

#### **Popular Feature Extraction Techniques**

1. **Principal Component Analysis ([[PCA]])** – Projects data into new principal components that explain the most variance.
2. **Linear Discriminant Analysis (LDA)** – Similar to PCA but focuses on maximizing class separability.
3. **t-SNE (t-Distributed Stochastic Neighbor Embedding)** – Non-linear method for visualization.
4. **Autoencoders (Neural Networks)** – Compress data using deep learning.

Example: Instead of using 3 features **(height, weight, BMI)**, PCA might extract 1 feature that represents **overall body size**.

---

## **PCA for Dimensionality Reduction**

PCA is the most common technique for dimensionality reduction. It finds new features (principal components) that retain most of the variance in the data.

- **Step 1**: Standardize the data (mean = 0, variance = 1).
- **Step 2**: Compute the covariance matrix.
- **Step 3**: Find eigenvalues & eigenvectors.
- **Step 4**: Select top kk components that explain most variance.
- **Step 5**: Transform data into new lower-dimensional space.

Example: If we have 100 features, PCA might reduce them to 2 or 3 while keeping most of the information.

---

## **Real-Life Examples of Dimensionality Reduction**

1. **Image Processing** – Reduce pixel dimensions while preserving important features.
2. **Text Data (NLP)** – Convert high-dimensional word embeddings into compact representations.
3. **Gene Expression Data** – Identify important genes from thousands of genetic markers.
4. **Customer Segmentation** – Reduce survey responses into key customer traits.

---
