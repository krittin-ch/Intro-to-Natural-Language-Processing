# https://towardsdatascience.com/a-guide-to-word-embeddings-8a23817ab60f

'''
Word Embeddings are dense representations of the individual words in a text, 
taking into account the context and other surrounding words that that individual word occurs with.

The dimensions of this real-valued vector can be chosen and the semantic relationships 
between words are captured more effectively than a simple Bag-of-Words Model.
'''

# Importing libraries
# Data Manipulation/ Handling
import pandas as pd, numpy as np

# Visualization
import seaborn as sb
import matplotlib.pyplot as plt

# NLP
import re
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess

# One Hot Encoder
from sklearn.preprocessing import OneHotEncoder as ohe
from sklearn.compose import ColumnTransformer

# Tokenization with Keras
from tensorflow.keras.preprocessing.text import Tokenizer
# Padding all questions with zeros
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten

stop_words = set(stopwords.words('english'))

path = 'C:/Users/kritt/Documents/GitHub/Intro-to-Natural-Language-Processing/word-embedding/'

# Importing dataset
# Index(['Id', 'Title', 'Body', 'Tags', 'CreationDate', 'Y'], dtype='object')
# Choose only columns = {'Body': 'question', 'Y': 'category'}
dataset = pd.read_csv(path + 'train.csv')[['Body', 'Y']].rename(columns = {'Body': 'question', 'Y': 'category'})
'''
         Id                                              Title  ...         CreationDate         Y
0  34552656             Java: Repeat Task Every Random Seconds  ...  2016-01-01 00:21:59  LQ_CLOSE
1  34553034                  Why are Java Optionals immutable?  ...  2016-01-01 02:03:20        HQ
2  34553174  Text Overlay Image with Darkened Opacity React...  ...  2016-01-01 02:48:24        HQ
3  34553318         Why ternary operator in swift is so picky?  ...  2016-01-01 03:30:17        HQ
4  34553755                 hide/show fab with scale animation  ...  2016-01-01 05:21:48        HQ

'''

# Index(['Id', 'Title', 'Body', 'Tags', 'CreationDate', 'Y'], dtype='object')
# Choose only columns = {'Body': 'question', 'Y': 'category'}
ds = pd.read_csv(path + 'valid.csv')[['Body', 'Y']].rename(columns = {'Body': 'question', 'Y': 'category'})
'''
         Id                                              Title  ...         CreationDate        Y
0  34552974  How to get all the child records from differen...  ...  2016-01-01 01:44:52  LQ_EDIT
1  34554721  Retrieve all except some data of the another t...  ...  2016-01-01 08:43:50  LQ_EDIT
2  34555135                                  Pandas: read_html  ...  2016-01-01 09:55:22       HQ
3  34555448                           Reader Always gimme NULL  ...  2016-01-01 10:43:45  LQ_EDIT
4  34555752    php rearrange array elements based on condition  ...  2016-01-01 11:34:09  LQ_EDIT
'''

# Cleaning up symbols & HTML tags
symbols = re.compile(pattern = '[/<>(){}\[\]\|@,;]')
tags = ['href', 'http', 'https', 'www']

def text_clean(s: str) -> str:
    s = symbols.sub(' ', s)
    for i in tags:
        s = s.replace(i, ' ')

    return ' '.join(word for word in simple_preprocess(s) if not word in stop_words)


# sentence = '<p>I am having an internship at BotNoi as a AI NLP<p>'
# text_clean(sentence)
# internship botnoi ai nlp

# Clean data
dataset.iloc[:, 0] = dataset.iloc[:, 0].apply(text_clean)
ds.iloc[:, 0] = ds.iloc[:, 0].apply(text_clean)


# Train & Test subsets
X_train, y_train = dataset.iloc[:, 0].values, dataset.iloc[:, 1].values.reshape(-1, 1)
X_test, y_test = ds.iloc[:, 0].values, ds.iloc[:, 1].values.reshape(-1, 1)

# Multi-class labeling by One Hot Encoder
# ['LQ_CLOSE' 'HQ' 'LQ_EDIT']
ct = ColumnTransformer(transformers = [('one_hot_encoder', ohe(categories = 'auto'), [0])],
                       remainder = 'passthrough')

y_train = ct.fit_transform(y_train)
y_test = ct.transform(y_test)

# Setting some paramters
vocab_size = 2000
sequence_length = 100

tk = Tokenizer(num_words = vocab_size)
tk.fit_on_texts(X_train)


X_train = tk.texts_to_sequences(X_train)
X_test = tk.texts_to_sequences(X_test)

X_train_seq = pad_sequences(X_train, maxlen = sequence_length, padding = 'post')
X_test_seq = pad_sequences(X_test, maxlen = sequence_length, padding = 'post')

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=5, input_length=sequence_length))
model.add(Flatten())

model.add(Dense(units = 3, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'rmsprop',
              metrics = ['accuracy'])

model.summary()

history = model.fit(X_train_seq, y_train, epochs = 20, batch_size = 512, verbose = 1)


# Evaluating model performance on test set
loss, accuracy = model.evaluate(X_test_seq, y_test, verbose = 1)
print("\nAccuracy: {}\nLoss: {}".format(accuracy, loss))

# Plotting Accuracy & Loss over epochs
sb.set_style('darkgrid')

# 1) Accuracy
plt.plot(history.history['accuracy'], label = 'training', color = '#003399')
plt.legend(shadow = True, loc = 'lower right')
plt.title('Accuracy Plot over Epochs')
plt.show()

# 2) Loss
plt.plot(history.history['loss'], label = 'training loss', color = '#FF0033')
plt.legend(shadow = True, loc = 'upper right')
plt.title('Loss Plot over Epochs')
plt.show()