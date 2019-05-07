from nltk import word_tokenize
from nltk.sentiment.util import mark_negation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.base import TransformerMixin
import pandas as pd
IMDB_data=pd.read_csv("labeledTrainData.tsv", delimiter = '\t', quoting = 3)
X=IMDB_data['review'].head(10000)
y=IMDB_data['sentiment'].head(10000)
X_list=[]
for i in X:
    X_list.append(i)
list_y=[]
for i in y:
    list_y.append(i)
    
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_list, list_y, test_size = 0.20, random_state = 0)

clf = Pipeline([
    ('vectorizer', CountVectorizer(analyzer="word",
                                   ngram_range=(2, 2),
                                   tokenizer=word_tokenize,
                                   preprocessor=lambda text: text.replace("<br />", " "),
                                   max_features=10000) ),
    ('classifier', LinearSVC())
])
 
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred)*100)#79.7%