# https://medium.com/@mervebdurna/advanced-word-embeddings-word2vec-glove-and-fasttext-26e546ffedbd

# Code Example with Toy Dataset
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Toy dataset
sentences = [
    "I love natural language processing.", 
    "Word embeddings are powerful."
]

# Tokenize sentences
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]

# Train Word2Vec
model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)

# Access Embedding
word_embedding = model.wv
print(word_embedding['natural'])