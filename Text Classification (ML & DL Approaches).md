---
tags: 
Date: "10-06-2025 18:33"
---

## Text Classification (ML & DL Approaches)

---

# Text Classification (ML & DL Approaches)

These notes summarize the principles of text classification, a fundamental supervised learning task in [[NLP]]. They cover both traditional Machine Learning (ML) and modern Deep Learning (DL) approaches.

> **Key Idea:** Text Classification is the process of assigning a piece of text to one or more predefined categories. The core challenge is to build a model that understands the text and makes accurate predictions.

### 1. Types of Text Classification

- **Binary Classification:** Assigning text to one of two mutually exclusive classes.
    
    - _Example:_ Email spam detection (spam vs. not spam).
        
- **Multi-class Classification:** Assigning text to one of more than two mutually exclusive classes.
    
    - _Example:_ Categorizing news articles into topics like 'Sports', 'Politics', or 'Business'.
        
- **Multi-label Classification:** Assigning text to one or more non-exclusive classes.
    
    - _Example:_ Tagging a single movie review with multiple labels like 'Action', 'Sci-Fi', and 'Adventure'.
        

### 2. Real-World Applications

- **Spam Detection:** The classic use case, filtering unwanted emails.
    
- **Customer Support:** Automatically routing support tickets to the correct department.
    
- **Sentiment Analysis:** Determining the sentiment of a text (positive, negative, neutral).
    
- **Language Detection:** Identifying the language of a piece of text.
    
- **Content Moderation:** Flagging and removing inappropriate content.
    
- **Fake News Detection:** Identifying misinformation.
    

### 3. The Text Classification Pipeline

A typical project follows these steps:

1. **Data Acquisition:** Gathering text data.
    
2. **Text Preprocessing:** Cleaning the text (lowercasing, removing stop words, etc.).
    
3. **Text Vectorization:** Converting text into numerical features.
    
4. **Modeling:** Training an algorithm on the features.
    
5. **Evaluation:** Assessing model performance (accuracy, precision, recall, F1-score).
    
6. **Deployment:** Making the model available for use.
    

### 4. Machine Learning Modeling Approaches

This approach involves using traditional ML algorithms with feature engineering.

#### a) Bag-of-Words (BoW)

- **Concept:** Represents text by the frequency of each word.
    
- **Implementation:** Using `CountVectorizer` from Scikit-learn.
    
- **Results from Video:**
    
    - With **Gaussian Naive Bayes:** ~60% accuracy.
        
    - With **Random Forest:** ~84% accuracy.
        
    - Using `max_features` to limit the vocabulary to the top 3000 most frequent words can improve performance and reduce memory usage.
        

#### b) N-grams

- **Concept:** An extension of BoW that considers sequences of N words, capturing some context.
    
- **Implementation:** Using the `ngram_range` parameter in `CountVectorizer` (e.g., `ngram_range=(1,2)` for unigrams and bigrams).
    
- **Note:** This can dramatically increase vocabulary size, so using `max_features` is often necessary to manage memory.
    

#### c) TF-IDF (Term Frequency-Inverse Document Frequency)

- **Concept:** A scoring method that gives higher weight to words that are frequent in a document but rare across the entire corpus.
    
- **Implementation:** Using `TfidfVectorizer` from Scikit-learn.
    
- **Results from Video:**
    
    - With **Random Forest:** Achieved a slightly better accuracy of ~85% compared to BoW.
        

#### d) Average Word2Vec

- **Concept:** Represents a whole document by taking the average of the Word2Vec vectors of all the words in it.
    
- **Implementation:**
    
    1. Train a Word2Vec model on your corpus (or use a pre-trained one).
        
    2. For each document, get the vector for every word and compute their average.
        
    3. Use these document vectors to train a classifier.
        
- **Results from Video:**
    
    - With **Random Forest:** Achieved ~70% accuracy. The lower performance was attributed to training the Word2Vec model on a small subset of the data. Performance would likely improve with more data.


### 5. Deep Learning Modeling Approaches

This is the state-of-the-art approach, using neural networks that automatically learn features from the text. The key difference is that the **Vectorization (Embedding)** and **Modeling** steps are combined.

#### a) Convolutional Neural Networks (CNNs) for Text

- **Concept:** Originally designed for images, CNNs can be applied to text by treating sentences as "images" where word embeddings are the pixels.
    
- **How it works:** Filters (kernels) of different sizes (e.g., 2-word, 3-word) slide over the text to capture local patterns (like n-grams). A pooling layer then extracts the most important features.
    
- **Pros:** Very fast and efficient at feature extraction.
    
- **Cons:** Does not explicitly model the sequential nature of language.
    

#### b) Recurrent Neural Networks (RNNs) - LSTMs & GRUs

- **Concept:** Designed specifically to handle sequential data like text. They process words one by one, maintaining a "memory" (hidden state) of what they have seen so far.
    
- **How it works:** An RNN reads the word embeddings sequentially. The final hidden state, which represents a summary of the entire sequence, is then used for classification.
    
- **Pros:** Captures long-range dependencies and the sequential nature of language.
    
- **Cons:** Can be slower to train than CNNs.
    

#### c) Transfer Learning & Pre-trained Models (BERT, etc.)

- **Concept:** This is the most powerful and current approach. It involves using a massive, pre-trained language model (like BERT, RoBERTa, or GPT) that has already learned the nuances of a language from billions of sentences.
    
- **How it works:**
    
    1. **Fine-Tuning:** You take a pre-trained model and re-train it (fine-tune) on your specific, smaller dataset. The model adapts its general language understanding to your particular classification task.
        
- **Pros:**
    
    - Achieves state-of-the-art performance with relatively little data.
        
    - Saves immense amounts of time and computational resources compared to training a model from scratch.
        
- **Implementation:** Using libraries like **Hugging Face's `transformers`** makes it easy to download and fine-tune these models.
    

### 6. Practical Advice

1. **Start Simple:** Always begin with a simple ML baseline (e.g., TF-IDF with Logistic Regression). Don't jump to deep learning unless necessary.
    
2. **Craft Features:** For ML models, combine automated features with hand-crafted, domain-specific ones if possible.
    
3. **Handle Imbalanced Datasets:** Use techniques like over-sampling (e.g., SMOTE) or under-sampling to prevent model bias if your classes are not balanced.
    
4. **Leverage Transfer Learning:** For high performance, fine-tuning a pre-trained model like BERT is often the best approach and should be preferred over training a deep learning model from scratch.
    
5. **Practice:** The best way to learn is by working on diverse text classification projects.