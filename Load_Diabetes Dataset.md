---
tags:
  - DL
  - workshops/pytorch
Date: 22-01-2025 11:08
---
[[Scikit-Learn]]  [[Machine Learning]]
## Load_Diabetes Dataset

---

The **load_diabetes dataset** is a built-in dataset in the `sklearn.datasets` module of the **scikit-learn** library. It contains data for a diabetes regression problem, which is useful for practicing machine learning, especially in [[regression]] tasks.

### Overview of the Diabetes Dataset:

- **Samples**: The dataset contains 442 samples.
- **Features**: There are 10 features (age, sex, BMI, blood pressure, and six blood serum measurements).
- **Target**: The target variable is a quantitative measure of disease progression one year after baseline.

### Features Description:

Each feature is a numeric value that has been normalized for simplicity:

1. Age
2. Sex
3. Body Mass Index (BMI)
4. Average Blood Pressure
5. Six blood serum measurements (S1, S2, S3, S4, S5, S6)

### How to Use the Diabetes Dataset:

#### 1. Importing the Dataset:

```python
from sklearn.datasets import load_diabetes
```

#### 2. Loading the Data:

You can load the dataset into memory using:

```python
diabetes = load_diabetes()
```

#### 3. Exploring the Data:

The dataset is a dictionary-like object containing the following keys:

- `data`: The features matrix (numpy array of shape `(442, 10)`).
- `target`: The target values (numpy array of shape `(442,)`).
- `feature_names`: The names of the features.
- `DESCR`: A description of the dataset.

Example of accessing the data:

```python
X = diabetes.data  # Features
y = diabetes.target  # Target
feature_names = diabetes.feature_names  # Feature names
```

#### 4. Splitting the Data:

You typically split the data into training and testing sets for model evaluation.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### 5. Applying a Regression Model:

For example, you can use a linear regression model:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

#### 6. Making Predictions:

After training, you can make predictions on the test data:

```python
y_pred = model.predict(X_test)
```

#### 7. Evaluating the Model:

Evaluate the model’s performance using metrics like Mean Squared Error (MSE):

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
```

### Summary of Steps:

1. Import and load the dataset.
2. Split the data into training and testing sets.
3. Choose a regression model (e.g., Linear Regression).
4. Train the model using the training data.
5. Predict and evaluate the model's performance on the test data.

This dataset provides a straightforward way to practice regression tasks, model evaluation, and feature exploration in machine learning.

The code:

```python
X_numpy, y_numpy = load_diabetes(return_X_y=True)
```

### Explanation:

- **`load_diabetes()`**: This function loads the diabetes dataset from `sklearn.datasets`.
- **`return_X_y=True`**: This parameter changes the default behavior of the function:
    - When `return_X_y=False` (default), the function returns the dataset as a dictionary-like object with keys like `data`, `target`, `DESCR`, etc.
    - When `return_X_y=True`, the function directly returns **only the features (X)** and **target (y)** as separate NumPy arrays.

### What the Code Does:

1. **`X_numpy`**:
    
    - Contains the feature matrix (all the input data).
    - Shape: `(442, 10)` (442 samples, 10 features).
    - Type: `numpy.ndarray`.
2. **`y_numpy`**:
    
    - Contains the target variable (the disease progression values for each sample).
    - Shape: `(442,)` (a 1D array with 442 values).
    - Type: `numpy.ndarray`.

By setting `return_X_y=True`, you're bypassing the other metadata (like `DESCR` or `feature_names`) and directly extracting the data and target in the form of NumPy arrays.

### Example of Use:

```python
from sklearn.datasets import load_diabetes

# Load features and target as NumPy arrays
X_numpy, y_numpy = load_diabetes(return_X_y=True)

print("Shape of X:", X_numpy.shape)  # (442, 10)
print("Shape of y:", y_numpy.shape)  # (442,)

# Perform any machine learning operations on X_numpy and y_numpy
```

### Benefits of `return_X_y=True`:

- **Convenience**: Avoids dealing with the dictionary-like structure.
- **Direct Input**: Provides features (`X`) and target (`y`) in a format directly usable by machine learning models in scikit-learn or other libraries.