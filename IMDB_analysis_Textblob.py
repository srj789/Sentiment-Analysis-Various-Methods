import pandas as pd
from textblob import TextBlob
from sklearn.metrics import accuracy_score

IMDB_data=pd.read_csv("labeledTrainData.tsv", delimiter = '\t', quoting = 3)

X=IMDB_data['review']
y=IMDB_data['sentiment']
list_X=[]
for i in X:
    list_X.append(i)
    
list_y=[]
for i in y:
    list_y.append(i)
    
def Textblob_polarity(text):
    score = TextBlob(text)
    score=score.sentiment.polarity
    #print(score)
    if (score>=0.1):
        return 1
    else:
        return 0
    
y_pred=[]    
for i in range(0,len(list_X)): 
    result=Textblob_polarity(list_X[i])
    y_pred.append(result)
    
print("Accuracy of Model : {}%".format(accuracy_score(y,y_pred)*100))#76.428%