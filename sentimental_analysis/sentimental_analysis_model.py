import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score

path = 'C:/Users/kritt/Documents/GitHub/Intro-to-Natural-Language-Processing/sentimental_analysis/'
dataset = pd.read_csv(path + 'a1_RestaurantReviews_HistoricDump.tsv', delimiter = '\t', quoting = 3)

'''
                                              Review  Liked
0                           Wow... Loved this place.      1
1                                 Crust is not good.      0
2          Not tasty and the texture was just nasty.      0
3  Stopped by during the late May bank holiday of...      1
4  The selection on the menu was great and so wer...      1
'''

# nltk.download('stopwords')
'''
all_stopwords = 
['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
"you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 
'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 
'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',
.
.
.
"isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 
'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
'''
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

ps = PorterStemmer()    # word stemming algorithm
corpus = []             # stemed sentences

for i in range(900):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

cv = CountVectorizer(max_features = 1420)   # Convert some words (max_features) to features
'''
For example, 
"I am a boy who is studying in university level as a higher education in electrical engineering at SIIT" 
with max_feature = 5  will be tokenize to

"is": 2
"a": 1
"boy": 1
"university": 1
"education": 1

[2,1,1,1,1]

Note: some words are excluded due to the being stopwords
'''

X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
print("\nAccuracy:", accuracy)

# # Save the classifier and CountVectorizer for future use
# joblib.dump(classifier, 'sentiment_classifier.joblib')
# pickle.dump(cv, open('count_vectorizer.pkl', 'wb'))


# Train the Logistic Regression classifier
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Predict the sentiment on the test set
y_pred = classifier.predict(X_test)

# Evaluate the performance of the classifier
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
print("\nAccuracy:", accuracy)

# # Save the classifier and CountVectorizer for future use
# joblib.dump(classifier, 'sentiment_classifier_logistic_regression.joblib')
# pickle.dump(cv, open('count_vectorizer.pkl', 'wb'))
