---
tags:
  - NLP
Date: 10-06-2025 18:10
---

## Text Vectorization for [NLP]

---

#  Text Vectorization (Feature Engineering)

> **Key Idea:** The primary challenge of text vectorization is to create numerical representations that successfully capture the **semantic meaning** of the text.

### Key Terminology

- **Corpus:** The entire collection of text documents in your dataset.
    
- **Vocabulary:** The set of all unique words in the corpus.
    
- **Document:** A single piece of text (e.g., a sentence, a review, an article).
    
- **Word/Term:** A single word in a document.
    

## 1. One-Hot Encoding (OHE)

- **What it is:** A method where each word is represented by a vector of the size of the vocabulary. The vector has a `1` at the index corresponding to that word and `0`s everywhere else.
    
- **Pros:**
    
    - Simple and intuitive to understand.
        
- **Cons:**
    
    - **Sparsity:** Creates extremely large and sparse vectors (mostly zeros), which is computationally inefficient.
        
    - **Variable Document Size:** Leads to representations of different sizes for documents with different lengths, which most ML models cannot handle.
        
    - **Out-of-Vocabulary (OOV):** Cannot handle words that were not in the training vocabulary.
        
    - **No Semantic Meaning:** Fails to capture any relationship between words. The vectors for "good" and "great" are as different as the vectors for "good" and "car".
        

## 2. Bag of Words (BoW)

- **What it is:** Represents each document as a vector of the size of the vocabulary. Each element in the vector contains the frequency (count) of the corresponding word from the vocabulary in that document.
    
- **Why it's done:** It's a simple way to create a fixed-size vector for each document while capturing some information about its content. The core idea is that documents with similar content will have similar word counts.
    
- **Pros:**
    
    - Simple to understand and implement.
        
    - Creates fixed-size vectors for each document.
        
- **Cons:**
    
    - **Loses Word Order:** Ignores grammar and the sequence of words. "This is not good" and "This is good" are seen as very similar.
        
    - **Sparsity:** Still produces sparse vectors, though less so than OHE.
        
    - **No Semantic Meaning:** Does not inherently understand synonyms or context.
        

## 3. N-grams (Bag of N-grams)

- **What it is:** An extension of Bag of Words where the "words" in the vocabulary are sequences of N consecutive words.
    
    - **Unigrams (N=1):** Standard BoW.
        
    - **Bigrams (N=2):** Pairs of words (e.g., "not good", "very happy").
        
    - **Trigrams (N=3):** Triplets of words.
        
- **Why it's done:** To capture some local word order and context that is lost in the standard BoW model. This helps in understanding phrases.
    
- **Pros:**
    
    - Captures context and some word order, which can significantly improve performance.
        
- **Cons:**
    
    - **Massively Increases Vocabulary Size:** Leads to very high-dimensional and sparse vectors, increasing computational cost.
        
    - Still suffers from the OOV problem.
        

## 4. TF-IDF (Term Frequency-Inverse Document Frequency)

- **What it is:** A statistical measure used to evaluate how important a word is to a document in a collection or corpus. It gives higher weight to words that are frequent in a specific document but rare across all documents.
    
- **How it works:**
    
    - **Term Frequency (TF):** Measures how often a word appears in a document. It's the count of a word in a document divided by the total number of words in that document.
        
    - **Inverse Document Frequency (IDF):** Measures how rare a word is across the entire corpus. It's the logarithm of (total number of documents / number of documents containing the word).
        
    - **TF-IDF Score = TF * IDF**.
        
- **Why it's done:** To down-weigh common words (like "the", "a") that appear everywhere and give more importance to words that are characteristic of a particular document.
    
- **Pros:**
    
    - Very effective at identifying which words are most representative of a document.
        
    - A standard and powerful tool in information retrieval and search engines.
        
- **Cons:**
    
    - Still produces sparse, high-dimensional vectors.
        
    - Does not capture semantic relationships (synonyms).
        
    - Loses word order (unless combined with N-grams).
        

## 5. Custom / Hand-Crafted Features

- **What it is:** Manually creating features based on domain knowledge about the text and the problem.
    
- **Why it's done:** To inject specific, task-relevant information into the model that might not be captured by automated methods.
    
- **Examples for Sentiment Analysis:**
    
    - Count of positive words.
        
    - Count of negative words.
        
    - Length of the document (word count, character count).
        
    - Ratio of positive to negative words.
        
- **Note:** These are often used in a **hybrid approach**, where they are combined with BoW or TF-IDF vectors to enrich the feature set.
    

> **Conclusion & Next Steps:** While these techniques are foundational, they all struggle to capture true semantic meaning. The next step in text vectorization, **Word Embeddings** (like Word2Vec, GloVe, and FastText), addresses this limitation by creating dense vectors where similar words are placed close to each other in the vector space.