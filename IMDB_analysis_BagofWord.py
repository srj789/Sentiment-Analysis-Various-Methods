import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

IMDB_data=pd.read_csv("labeledTrainData.tsv", delimiter = '\t', quoting = 3)

corpus = []
for i in range(0, 1000):
    result = re.sub('[^a-zA-Z]', ' ', IMDB_data['review'][i])
    result = result.lower()
    result = result.split()
    ps = PorterStemmer()
    result = [ps.stem(word) for word in result if not word in set(stopwords.words('english'))]
    result = ' '.join(result)
    corpus.append(result)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 500)
X = cv.fit_transform(corpus).toarray()
y = IMDB_data.iloc[0:1000, 1].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#applying gaussianNB model
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#applying multinomial model
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

classifier.score(X_train,y_train)

y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred)*100)




