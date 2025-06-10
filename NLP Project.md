---
tags: 
Date: "10-06-2025 18:48"
---

## [NLP] Project

---

# Project Summary: Quora Duplicate Question Pairs

This document summarizes the end-to-end data science project presented in the video, which focuses on identifying whether a pair of questions on Quora are duplicates (i.e., have the same intent).

> **Problem Statement:** To build a model that can predict if two questions are duplicates. This is a **binary classification** task, where the output is either `1` (duplicate) or `0` (not a duplicate).

### 1. Dataset Overview

- **Source:** Quora Question Pairs dataset (available on Kaggle).
    
- **Content:** Contains pairs of questions and a label (`is_duplicate`) indicating if they are duplicates.
    
- **Size:** Over 400,000 question pairs.
    

### 2. Exploratory Data Analysis (EDA)

The initial analysis of the dataset revealed several key insights:

- **Class Distribution:** The dataset is imbalanced.
    
    - ~63% of pairs are **not duplicates** (label 0).
        
    - ~37% of pairs are **duplicates** (label 1).
        
- **Question Analysis:**
    
    - A total of 537,933 unique questions are present in the dataset.
        
    - Some questions appear multiple times, with the most frequent question appearing 157 times.
        

### 3. Feature Engineering

This is the most critical part of the project. Since machine learning models can't work with raw text, a set of meaningful numerical features were created from the question pairs.

#### a) Basic Features

- `q1_len`: Length of question 1.
    
- `q2_len`: Length of question 2.
    
- `q1_num_words`: Number of words in question 1.
    
- `q2_num_words`: Number of words in question 2.
    
- `word_common`: Number of common words between the two questions.
    
- `word_total`: Total number of words in both questions.
    
- `word_share`: Ratio of common words to total words (`word_common` / `word_total`). This is a key feature indicating similarity.
    

#### b) Advanced "Fuzzy" Features

To capture more nuanced similarity, the `fuzzywuzzy` library was used to generate features based on string matching algorithms:

- `fuzz_ratio`: Measures the simple ratio of similarity between the two strings.
    
- `fuzz_partial_ratio`: Measures similarity based on the best matching sub-string.
    
- `token_sort_ratio`: Measures similarity after tokenizing and sorting the words alphabetically.
    
- `token_set_ratio`: An advanced ratio that handles cases where sentences are of different lengths but contain similar words.
    

### 4. Model Building and Evaluation

After creating the features, various machine learning models were trained and evaluated.

- **Approach:** The engineered features (not the raw text) were used as input to the models.
    
- **Algorithms Used:** The project focused on tree-based ensemble models, which are powerful for tabular data.
    
    - **Random Forest:** A strong baseline model.
        
    - **XGBoost (Extreme Gradient Boosting):** Often the top-performing algorithm for classification tasks on structured data.
        
- **Evaluation Metric:** Due to the class imbalance, **Log-Loss** was used as the primary evaluation metric, as it penalizes models that are confidently wrong. Accuracy alone would be misleading.
    

### 5. Building a Web Application

To demonstrate the final model, a simple web application was built using **Streamlit**.

- **Functionality:**
    
    1. The user enters two questions into input boxes.
        
    2. The application performs the same feature engineering steps on the input.
        
    3. The trained XGBoost model predicts whether the pair is a duplicate.
        
    4. The result (Duplicate or Not Duplicate) is displayed to the user.
        
- **Significance:** This step shows how to take a trained model from a notebook environment and deploy it into a usable application for end-users.