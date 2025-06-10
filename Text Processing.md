---
tags:
  - NLP
Date: 10-06-2025 18:06
---

## Text processing for [[NLP]]

---

### 1. Lowercasing

- **What it is:** Converting all characters in the text to lowercase.
    
- **Why it's done:** To prevent the model from treating the same word with different capitalization as distinct entities (e.g., "Apple" the company vs. "apple" the fruit, or "The" at the start of a sentence vs. "the" in the middle). This reduces the vocabulary size and simplifies the model's task.
    
- **How it's done:** Using the `.lower()` string method in Python.
    

### 2. Removing HTML Tags

- **What it is:** Stripping out all HTML markup (e.g., `<p>`, `<h1>`, `<b>`).
    
- **Why it's done:** HTML tags are used for rendering content in browsers and provide no linguistic value for most NLP tasks. They are noise that can confuse the model.
    
- **How it's done:** Using regular expressions (regex) with Python's `re` library to find and remove patterns that look like HTML tags.
    

### 3. Removing URLs

- **What it is:** Deleting web addresses (URLs) from the text.
    
- **Why it's done:** Like HTML tags, URLs typically don't contribute to the semantic meaning of the text for tasks like sentiment analysis and can add unnecessary noise.
    
- **How it's done:** Using regular expressions to identify and remove various URL formats (e.g., those starting with `http://`, `https://`, or `www`).
    

### 4. Removing Punctuation

- **What it is:** Eliminating punctuation marks like `!`, `?`, `.`, `,`.
    
- **Why it's done:** Punctuation can be problematic during tokenization, where a word might be split incorrectly (e.g., "hello!" might become a different token from "hello"). Removing it helps standardize the text.
    
- **How it's done:**
    
    - **Efficient Method:** Using Python's `str.maketrans` and `translate` functions for fast removal.
        
    - **Less Efficient Method:** Looping through each character and checking if it's in a list of punctuation marks (e.g., from `string.punctuation`).
        

### 5. Handling Chat Words (Slang/Shortforms)

- **What it is:** Expanding common chat abbreviations and slang into their full forms.
    
- **Why it's done:** Models are trained on standard language and won't understand slang (e.g., "GN" for "Good Night", "LOL" for "Laughing Out Loud"). Expanding them makes the text more standard.
    
- **How it's done:** Creating a dictionary (key-value pairs) that maps the slang to its expansion and then replacing the words in the text.
    

### 6. Spelling Correction

- **What it is:** Identifying and correcting spelling mistakes.
    
- **Why it's done:** Misspelled words (e.g., "awesom" vs. "awesome") increase the vocabulary size unnecessarily and can cause the model to miss the word's true meaning.
    
- **How it's done:** Using external libraries like **`textblob`** in Python, which provides a simple `.correct()` method.
    

### 7. Removing Stop Words

- **What it is:** Removing common words that have little semantic value on their own.
    
- **Why it's done:** Stop words (e.g., "a", "an", "the", "is", "in", "on") are frequent but often don't help in tasks like sentiment analysis or document classification. Removing them helps the model focus on the more important words.
    
- **How it's done:** Using a predefined list of stop words from libraries like **NLTK** (`nltk.corpus.stopwords`) and filtering them out from the tokenized text.
    
- **Caution:** For some tasks like machine translation or Part-of-Speech tagging, stop words are essential and should not be removed.
    

### 8. Handling Emojis

- **What it is:** Dealing with emojis present in the text.
    
- **Why it's done:** Most NLP models cannot process emojis directly.
    
- **How it's done:**
    
    1. **Remove Emojis:** Use regular expressions to find and delete emojis from the text.
        
    2. **Replace with Text:** Convert emojis into their textual descriptions (e.g., 😊 becomes `:grinning_face:`). This can preserve the emotional sentiment. The **`emoji`** library in Python is excellent for this.
        

### 9. Tokenization

- **What it is:** The process of breaking down a piece of text into smaller units called "tokens" (which can be words, sentences, or sub-words).
    
- **Why it's done:** It's the fundamental step for turning text into data that can be used for feature engineering. Models work with tokens, not whole strings.
    
- **Methods & Comparison:**
    
    - **`.split()` method:** Very basic, fails on complex cases with varied punctuation.
        
    - **Regular Expressions:** More powerful but can be complex to write a robust pattern.
        
    - **NLTK (`word_tokenize`):** A popular library that handles many common edge cases well.
        
    - **spaCy:** Widely considered the gold standard for tokenization due to its high speed and accuracy, as it's designed for production environments.
        

### 10. Stemming and Lemmatization

- **What it is:** Techniques to reduce words to their root form.
    
- **Why it's done:** To group different inflections of a word as a single item, reducing vocabulary size and helping the model recognize that words like "run", "running", and "ran" are related.
    
- **Stemming:**
    
    - **How:** A crude, rule-based process of chopping off word endings (e.g., -ing, -ed).
        
    - **Result:** Fast, but the resulting "stem" may not be a real word (e.g., "studies" -> "studi").
        
    - **Use Case:** Good for speed-critical applications like information retrieval where the output doesn't need to be human-readable.
        
    - **Example:** `PorterStemmer` from NLTK.
        
- **Lemmatization:**
    
    - **How:** A more sophisticated, dictionary-based process that returns the root form of the word (the "lemma").
        
    - **Result:** Slower, but always produces a valid, dictionary-lookup word. It is more linguistically accurate.
        
    - **Use Case:** Better for applications where meaning is critical, like chatbots or question-answering systems.
        
    - **Example:** `WordNetLemmatizer` from NLTK, which often requires a Part-of-Speech tag for accuracy.