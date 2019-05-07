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
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = IMDB_data.iloc[0:1000, 1].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

import keras
from keras.models import Sequential
from keras.layers import Dense
# Initialising the ANN
classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 1500))
# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

y_pred = classifier.predict(X_test)

y_list=[]
for i in range(0,len(y_pred)):
    if(y_pred[i]>0.5):
        y_list.append(1)
    else:
        y_list.append(0)
    
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_list)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_list)*100)     #84%


