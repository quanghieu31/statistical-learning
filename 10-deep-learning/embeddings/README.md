
/* *Digitized notes from class *   
Jan 14, 2025

1, Feature-based
Traditional way
"Smooth" features: differential features with some well-defined analytical characteristics

Kernel function $f()$
$f(\text{data}) = \text{embeddings}$ 
Goal: find a correct/make-sense function for embeddings
$K(x_1, x_2) = <\phi(x_1), \phi(x_2)>$

2, Bag-of-words
==what does it mean by "terms" and "terms" vs "tokens"== 

"Terms are commonly single words separated by whitespace or punctuation on either side (a.k.a. unigrams). In such a case, this is also referred to as "bag of words" representation because the counts of individual words is retained, but not the order of the words in the document." (Wiki on doc-term matrix)

Typically, 20,000-30,000 dimensions/unique words in any corpus on average (for "large" ones)
but these vectors are sparse!
something like this **document-term matrix**

|       | word_1 | word_2 | word_3 | ... | word_30000 |
| ----- | ------ | ------ | ------ | --- | ---------- |
| doc_1 | 2      | 0      | 0      |     | 0          |
| doc_2 | 0      | 1      | 0      |     | 0          |
| doc_3 | 0      | 0      | 0      |     | 1          |
"While the value of the cells is commonly the raw count of a given term, there are various schemes for weighting the raw counts such as row normalizing (i.e. relative frequency/proportions) and tf-idf." (Wiki on doc-term matrix)

But these are not enough for embeddings, we want each word to have unique embedding!

3, More learnable embeddings (using the counting/BOW/doc-term matrix above)

A dimensionality problem: each word in our document is represented by a one-hot-encoded vector (dummy variable) with 10,000 elements (one per word in the dictionary)! An approach that has become popular is to represent each word in a much lower-dimensional embedding embedding space. This means that rather than representing each word by a binary vector with 9,999 zeros and a single one in some position, we will represent it instead by a set of m real numbers, none of which are typically zero. Here m is the embedding dimension, and can be in the low 100s, or even less.
This means (in our case) that we need a matrix E of dimension m×10,000, where each column is indexed by one of the 10,000 words in our dictionary, and the values in that column give the m coordinates for that word in the embedding space (like this )

| m    | word1 | word2 | w3   | ... | w10000 |
| ---- | ----- | ----- | ---- | --- | ------ |
| m1   | 0.12  | 0.40  | 0.12 |     | 0.40   |
| m2   | 0.23  | 0.23  | 0.23 |     | 0.23   |
| m3   | 0.95  | 0.99  | 0.95 |     | 0.99   |
| ...  |       |       |      |     |        |
| m100 | 0.63  | 0.10  | 0.63 |     | 0.10   |

Some popular embedding methods

3.1, **TD-IDF**

Consider three documents:
- Document 1: "Data science is science."
- Document 2: "I love data."
- Document 3: "Science is fun, and data science is powerful."

Step 1: Vocabulary
`["data", "science", "is", "love", "fun", "and", "powerful"]`

Term Frequencies (TF)  // the basic document-term matrix (counting)
Counts the number of times a word appears in each document.

|           | data | science | is  | love | fun | and | powerful |
| --------- | ---- | ------- | --- | ---- | --- | --- | -------- |
| **Doc 1** | 1    | 2       | 1   | 0    | 0   | 0   | 0        |
| **Doc 2** | 1    | 0       | 0   | 1    | 0   | 0   | 0        |
| **Doc 3** | 2    | 1       | 2   | 0    | 1   | 1   | 1        |

Step 2: Inverse Document Frequency (IDF)  
Formula:  
$$\text{IDF}(t) = \log_{10}\left(\frac{N}{1 + \text{DF}(t)}\right)$$
where $N$ is the total number of documents, and $\text{DF}(t)$ is the number of documents containing term $t$. TF-IDF is computed for each document individually but relies on **global statistics** across the corpus to determine the importance of terms

| Term     | DF  | IDF (logarithmic base 10)                             |
| -------- | --- | ----------------------------------------------------- |
| data     | 3   | $\log_{10}\left(\frac{3}{1 + 3}\right) \approx 0$     |
| science  | 2   | $\log_{10}\left(\frac{3}{1 + 2}\right) \approx 0.124$ |
| is       | 2   | $\log_{10}\left(\frac{3}{1 + 2}\right) \approx 0.124$ |
| love     | 1   | $\log_{10}\left(\frac{3}{1 + 1}\right) \approx 0.301$ |
| fun      | 1   | $\log_{10}\left(\frac{3}{1 + 1}\right) \approx 0.301$ |
| and      | 1   | $\log_{10}\left(\frac{3}{1 + 1}\right) \approx 0.301$ |
| powerful | 1   | $\log_{10}\left(\frac{3}{1 + 1}\right) \approx 0.301$ |

Step 3: TF-IDF Values  
(TF multiplied by IDF)

| Document  | data | science | is    | love  | fun   | and   | powerful |
| --------- | ---- | ------- | ----- | ----- | ----- | ----- | -------- |
| Doc 1 | 0    | 0.248   | 0.124 | 0     | 0     | 0     | 0        |
| Doc 2 | 0    | 0       | 0     | 0.301 | 0     | 0     | 0        |
| Doc 3 | 0    | 0.124   | 0.248 | 0     | 0.301 | 0.301 | 0.301    |

- The term "data" has an IDF VERY close to 0 because it appears in every document, making it less informative.
- Rare terms like "love," "fun," and "powerful" have higher IDF values, increasing their importance in the TF-IDF matrix.


3.2, Lower dimension with PCA, SVD like LSA

3.3, Word2vec (CBOW, skip-gram) - Maps words to dense vector representations based on their surrounding context.  

3.4, Clin2vec - domain-specific BOW like insurance claims or cui2vec

3.5, Universal chemical key (standardization attempt for "SMILES strings" genes)

4, Normalization methods
Very important
- L2 norm (Euclidean): each embedding is divided by the sum of all embedding distances (each from 0)
	- Prevents longer documents from dominating due to higher term counts and ensures that the cosine similarity is bounded between -1 and 1.
- L1 norm (Manhantan)
	- Useful for sparsity-preserving models and when comparing weighted frequency distributions.
- Max norm (Divides each TF-IDF score by the maximum score in the document)
	- Ensures all values in the vector are scaled between 0 and 1 while preserving the relative importance of terms within a document.


---

1. Statistical Representations:  
   - Document-Term Matrix (DTM): Represents documents based on raw counts of words.  
   - TF-IDF Matrix: Weights terms based on their importance across the corpus.  
   - Co-occurrence Matrices: Captures how frequently terms appear together in context windows.  

2. Dense Word Embeddings (Shallow Neural Networks):  
   - Word2Vec (CBOW, Skip-gram): Maps words to dense vector representations based on their surrounding context.  
   - GloVe (Global Vectors): Learns word embeddings by factorizing word co-occurrence matrices.  
   - FastText: Extends Word2Vec by using character n-grams, making it robust to rare or misspelled words.  

3. Contextual Embeddings (Deep Learning-Based):  
   - ELMo (Embeddings from Language Models): Contextual word embeddings generated by a bidirectional LSTM language model.  
   - BERT (Bidirectional Encoder Representations from Transformers): Creates contextual embeddings based on the entire sentence, improving downstream NLP tasks.  
   - GPT Series: Embeds tokens based on autoregressive, transformer-based predictions of future words.  

4. Sentence and Document-Level Embeddings:  
   - Doc2Vec: Extends Word2Vec to produce embeddings for entire sentences or documents.  
   - Sentence-BERT (SBERT): Generates embeddings for entire sentences using transformer-based models optimized for semantic similarity.  

5. Hybrid or Task-Specific Representations:  
   - Latent Semantic Analysis (LSA): Reduces the dimensionality of a document-term matrix using singular value decomposition (SVD).  
   - Topic Modeling (LDA): Generates embeddings based on the topic distributions of documents.  
   - Graph-based Representations: Uses graph embeddings (like Node2Vec) to embed text documents linked by a relationship (e.g., citations, hyperlinks).  


---
