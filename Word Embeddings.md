---
tags: 
Date: "10-06-2025 18:13"
---

## Word Embeddings

---

# Word Embeddings (Word2Vec)

These notes summarize the video on Word Embeddings, a set of modern techniques for text vectorization. These methods address the major limitations of frequency-based approaches like Bag of Words and TF-IDF, particularly their inability to capture true semantic meaning.

> **Key Idea:** Word Embeddings represent words as dense, low-dimensional vectors in such a way that words with similar meanings have similar vector representations.

### The Core Idea: Distributional Semantics

Word Embeddings are built on a simple yet powerful linguistic concept: the **Distributional Hypothesis**.

> "You shall know a word by the company it keeps." - J.R. Firth

This means that words that appear in similar contexts (i.e., surrounded by similar words) tend to have similar meanings. For example, "cat" and "dog" often appear in sentences with words like "pet", "food", "leash", and "house". Word2Vec is designed to learn these relationships from a large corpus of text.

## Word2Vec: The State of the Art

Word2Vec is a popular model developed at Google to create word embeddings. It's not just one algorithm, but a family of model architectures.

### Advantages over Traditional Methods (BoW/TF-IDF)

1. **Captures Semantic Meaning:** It understands that words like "happy" and "joyful" are similar, which BoW/TF-IDF cannot do.
    
2. **Dense Vectors:** It produces dense vectors (most values are non-zero), which are more efficient than the sparse vectors from older methods.
    
3. **Lower Dimensionality:** The resulting vectors are much lower in dimension (typically 50-300 dimensions) compared to the vocabulary-sized vectors of BoW/TF-IDF, which reduces computational complexity.
    

### The "Fake Task" Approach

Word2Vec doesn't learn the vectors directly. Instead, it trains a shallow neural network on a "fake" supervised task. The actual word embeddings are a **by-product** of this training process—they are the learned weights of the hidden layer.

## Word2Vec Architectures

There are two main architectures for training a Word2Vec model:

### 1. CBOW (Continuous Bag of Words)

- **The "Fake Task":** Predict a target (middle) word based on its surrounding context words.
    
- **How it Works:** The model takes the context words as input, averages their vectors, and tries to predict the word that belongs in the middle.
    
- **Best For:**
    
    - Faster to train.
        
    - Performs better for frequent words.
        

### 2. Skip-Gram

- **The "Fake Task":** Predict the surrounding context words based on a given target (middle) word.
    
- **How it Works:** The model takes a single word as input and tries to predict the words that appear in its context window.
    
- **Best For:**
    
    - Works well with smaller amounts of training data.
        
    - Better at representing rare words and phrases.
        

## The Power of Learned Embeddings: Vector Arithmetic

Because the learned vectors capture meaning, they also capture relationships between words. This allows for powerful "vector arithmetic."

- **Similarity:** You can calculate the cosine similarity between vectors to find how "close" two words are in meaning.
    
    - `similarity('king', 'queen')` -> High value
        
    - `similarity('king', 'apple')` -> Low value
        
- **Analogy:** You can perform mathematical operations on the vectors to solve analogies.
    
    - `vector('king') - vector('man') + vector('woman') ≈ vector('queen')`
        
    - This works because the vector difference between 'king' and 'man' captures the concept of "royalty," which is then added to 'woman'.
        

### Practical Considerations

- **Implementation:** Word2Vec can be easily implemented using libraries like **`gensim`** in Python.
    
- **Training vs. Pre-trained:**
    
    - You can train your own Word2Vec model on a domain-specific corpus (e.g., medical texts, legal documents).
        
    - Alternatively, you can use **pre-trained models** (like Google News, trained on billions of words) which have already learned general language relationships.
        
- **Improving Quality:** The quality of embeddings can be improved by:
    
    - Using more training data.
        
    - Increasing the vector dimensions (e.g., from 100 to 300).
        
    - Adjusting the context window size.
        

> **Conclusion:** Word Embeddings like Word2Vec represent a major leap forward from simple frequency counts. By learning from context, they create rich, dense representations of words that capture semantic meaning, enabling more sophisticated and nuanced [NLP] applications.