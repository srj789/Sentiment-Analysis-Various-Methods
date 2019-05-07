import pandas as pd
from sklearn.metrics import accuracy_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

IMDB_data=pd.read_csv("labeledTrainData.tsv", delimiter = '\t', quoting = 3)
# checking for the any missing or null value
IMDB_data.isnull().values.any()
X=IMDB_data['review']
y=IMDB_data['sentiment']
list_X=[]
for i in X:
    list_X.append(i)
    
list_y=[]
for i in y:
    list_y.append(i)

vader = SentimentIntensityAnalyzer()
def vader_polarity(text):
    score = vader.polarity_scores(text)
    if score['pos'] > score['neg']:
        return 1
    else:
        return 0
y_pred=[]    
for i in range(0,len(list_X)): 
    result=vader_polarity(list_X[i])
    y_pred.append(result)
    
print("Accuracy of Model : {}%".format(accuracy_score(y,y_pred)*100))#69.108%


