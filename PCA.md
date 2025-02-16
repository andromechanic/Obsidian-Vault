---
tags:
  - IITM/MLT
Date: 16-02-2025 13:05
back links: "[[PCA Material]]"
---

## PCA

---

### **Principal Component Analysis (PCA)**

PCA (Principal Component Analysis) is a [[Dimensionality Reduction]] technique used in machine learning and statistics. It helps simplify complex datasets by reducing the number of features (dimensions) while preserving as much important information (variance) as possible.

---

## **Why Do We Need PCA?**

1. **[[Curse of Dimensionality]]** – When we have too many features, models become slow, overfit, and hard to interpret.
2. **Data Visualization** – PCA helps in reducing high-dimensional data (e.g., 100 features) into 2D or 3D for visualization.
3. **Noise Reduction** – By focusing on the most important patterns in data, PCA can remove noise.
4. **Better Model Performance** – Reducing redundant features can improve machine learning model efficiency.

---

## **How Does PCA Work?**

PCA works by transforming data into a new coordinate system such that the greatest variance comes on the first coordinate (Principal Component 1), the second greatest variance on the second coordinate (Principal Component 2), and so on.

### **Steps Involved in PCA**

1. **Standardize the Data** – Since PCA is affected by the scale of data, we standardize features (subtract mean, divide by standard deviation).
2. **Compute Covariance Matrix** – This captures relationships between variables (features).
3. **Compute Eigenvalues & Eigenvectors** – Eigenvalues tell us how much variance each component explains. Eigenvectors define the directions of new principal components.
4. **Sort and Select Principal Components** – We choose top components that explain most of the variance.
5. **Transform the Data** – The original data is projected onto the new principal components.

---

## **Mathematics Behind PCA**

### **1. Standardization**

Each feature in the dataset is transformed as:

$$
X′= \frac{X - \text{mean}(X)}{\text{std}(X)}
$$

This ensures all features have the same scale.

### **2. Covariance Matrix**

The covariance between two variables $X_i$ and $X_j$ is:

$$
Cov(Xi,Xj) =  \frac{1}{n-1} \sum_{k=1}^{n} (X_{ki} - \bar{X_i})(X_{kj} - \bar{X_j})
$$

This tells us if variables move together (positive covariance) or in opposite directions (negative covariance).

### **3. Eigenvalues & Eigenvectors**

For the covariance matrix CC, we solve:

$Cv = \lambda v$

- $v$ (Eigenvectors) define new feature directions.
- $λ$ (Eigenvalues) tell how much variance each component explains.

### **4. Selecting Principal Components**

We sort eigenvalues in descending order and pick the top $k$ components.

### **5. Project Data onto New Basis**

Final transformation:

$Z= X W$

where $W$ is the matrix of selected eigenvectors.

---

## **Where is PCA Used in Machine Learning?**

1. **Face Recognition** – PCA extracts important facial features.
2. **Handwriting Digit Recognition** – PCA reduces dimensions in MNIST dataset.
3. **Stock Market Analysis** – PCA finds patterns in financial data.
4. **Medical Diagnosis** – PCA reduces complex patient data into meaningful components.

---

## **Example: PCA in Action**

Imagine a dataset with **height and weight** as features. PCA might find:

- **PC1**: Overall body size (combination of height and weight).
- **PC2**: The difference between height and weight.

By reducing data from 2D to 1D (PC1), we still retain most of the information.

---

