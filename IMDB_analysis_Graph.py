import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

IMDB_data=pd.read_csv("labeledTrainData.tsv", delimiter = '\t', quoting = 3)
# checking for the any missing or null value
IMDB_data.isnull().values.any()
X=IMDB_data['review'].head(20)
y=IMDB_data['sentiment'].head(20)
z=IMDB_data['id'].head(20)
z1=[]
for i in z:
    z1.append(i)
score_list=[]
vader = SentimentIntensityAnalyzer()
def vader_polarity(text):
    score = vader.polarity_scores(text)
    print(score)
    score_list.append(score['pos']*100)
    if score['pos'] > score['neg']:
        return 1
    else:
        return 0
y_pred=[]    
for i in range(0,len(X)): 
    result=vader_polarity(X[i])
    y_pred.append(result)
print(score_list)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#left = [1, 2, 3, 4, 5,6,7,8,9,10]
plt.bar(z1,score_list, tick_label = z1, 
        width = 0.4, color = 'green')
plt.xticks(z1, z1, rotation='vertical')
plt.title("IMDB Movie Review")
plt.ylabel('Review Positive Polarity')
plt.xlabel('Movie ID')
plt.show()