---
tags:
  - ML
Date: 17-02-2025 08:41
---

## Kernel PCA

---

# **Kernel PCA (Principal Component Analysis)**

Kernel PCA is an extension of **Principal Component Analysis (PCA)** that allows it to work with **nonlinear data**. It uses the **[[Kernel Trick]]** to map data into a higher-dimensional space where it can be separated linearly.

---

## **Why Do We Need Kernel PCA?**

Regular **[[PCA]]** is a **linear technique**, meaning it works well when data can be separated by a straight line or plane. However, in real-world problems, data is often **nonlinear**.

🔹 **Example: Spiral Dataset**

- If data is arranged in a spiral shape, **regular PCA fails** because it can only capture linear patterns.
- **Kernel PCA transforms the data** into a higher-dimensional space where it can be separated more effectively.

### **When to Use Kernel PCA?**

- When your data is **nonlinear** and PCA does not capture important patterns.
- When performing **dimensionality reduction** for clustering, classification, or visualization.
- When you want a more flexible version of PCA that adapts to different types of data.

---

## **How Does Kernel PCA Work?**

Instead of directly applying PCA, **Kernel PCA** does the following:

### **Step 1: Choose a Kernel Function**

A **kernel function** is used to transform data into a higher-dimensional space. Common kernel functions:

1. **Polynomial Kernel**: $K(x,y)= (x \cdot y + c)^d$
    - Captures polynomial relationships.
2. **Gaussian (RBF) Kernel**: $K(x,y)= \exp \left( -\frac{\| x - y \|^2}{2\sigma^2} \right)$
    - Captures highly complex, nonlinear patterns.
3. **Sigmoid Kernel**: $K(x,y)=\tanh( \alpha x \cdot y + c)$
    - Similar to neural networks' activation functions.

### **Step 2: Compute Kernel Matrix**

The **kernel matrix (Gram matrix)** is computed using the chosen kernel function. Instead of explicitly mapping data to a higher dimension, we compute the similarities between data points in the transformed space.

### **Step 3: Apply PCA**

Once the kernel matrix is computed, we apply PCA as usual:

- Compute **eigenvalues and eigenvectors**.
- Select the top **k principal components**.
- Transform data into the new lower-dimensional space.

---

## **Mathematics Behind Kernel PCA**

1. **Compute Kernel Matrix $K$** using the kernel function: $Kij= K(x_i, x_j)$
2. **Center the Kernel Matrix**: $K′=K - \mathbf{1}K - K\mathbf{1} + \mathbf{1}K\mathbf{1}$
3. **Find Eigenvalues and Eigenvectors** of $K′$.
4. **Select Top $k$ Eigenvectors** for dimensionality reduction.

---

## **Kernel PCA vs. Standard PCA**

|Feature|Standard PCA|Kernel PCA|
|---|---|---|
|Type|Linear|Nonlinear|
|Transformation|Projects onto linear components|Uses kernel trick for nonlinear mapping|
|When to Use|When data is already linear|When data has complex, nonlinear patterns|
|Example|Handwritten digit recognition with clear features|Spiral dataset, complex patterns|

---

## **Real-Life Applications of Kernel PCA**

✅ **Image Recognition** – Extracts complex patterns from images.  
✅ **Facial Recognition** – Captures nonlinear relationships in facial features.  
✅ **Medical Diagnosis** – Identifies hidden patterns in genetic or medical data.  
✅ **Anomaly Detection** – Finds outliers in financial fraud detection.

---

## **Kernel PCA(Example)**

```python
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_moons

# Generate nonlinear dataset (moon-shaped)
X, _ = make_moons(n_samples=200, noise=0.05, random_state=42)

# Apply Kernel PCA with RBF kernel
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=10)
X_kpca = kpca.fit_transform(X)

# Plot results
plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c='b', alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Kernel PCA with RBF Kernel")
plt.show()

```

