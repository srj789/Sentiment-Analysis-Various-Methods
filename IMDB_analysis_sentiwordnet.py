from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from sklearn.metrics import accuracy_score
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag
import pandas as pd
import nltk
nltk.download('sentiwordnet')
  
IMDB_data=pd.read_csv("labeledTrainData.tsv", delimiter = '\t', quoting = 3)
X=IMDB_data['review']
y=IMDB_data['sentiment']
X_list=[]
for i in X:
    X_list.append(i)
list_y=[]
for i in y:
    list_y.append(i)

lemmatizer = WordNetLemmatizer()
def penn_to_wn(tag):
    """Convert between the PennTreebank tags to simple Wordnet tags
    """
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None

def clean_text(text):
    text = text.replace("<br />", " ")
    return text
def swn_polarity(text):
    """Return a sentiment polarity: 0 = negative, 1 = positive
    """
    sentiment = 0.0
    tokens_count = 0
    text = clean_text(text)
    
    raw_sentences = sent_tokenize(text)
    for raw_sentence in raw_sentences:
        tagged_sentence = pos_tag(word_tokenize(raw_sentence))
 
        for word, tag in tagged_sentence:
            wn_tag = penn_to_wn(tag)
            if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
                continue
 
            lemma = lemmatizer.lemmatize(word, pos=wn_tag)
            if not lemma:
                continue
 #find synonyms sense
            synsets = wn.synsets(lemma, pos=wn_tag)
            if not synsets:
                continue
 # Take the first sense, the most common
            synset = synsets[0]
            swn_synset = swn.senti_synset(synset.name())
 
            sentiment += swn_synset.pos_score() - swn_synset.neg_score()
            tokens_count += 1
# judgment call ? Default to positive or negative
    if not tokens_count:
        return 0
  # sum greater than 0 => positive sentiment
    if sentiment >= 0:
        return 1
  # negative sentiment
    return 0
y_pred=[]
for i in range(0,len(X_list)):
    score=swn_polarity(X_list[i])
    y_pred.append(score)
    
print("Accuracy of Model : {}%".format(accuracy_score(list_y,y_pred)*100))#60%
    
    
 
 
 