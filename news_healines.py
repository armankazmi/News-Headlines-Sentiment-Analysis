#Sentiment Analysis on Latest News Headlines
#author : Arman Kazmi

#Importing the Libraries
from gensim.parsing.preprocessing import remove_stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from textblob import TextBlob
import re


#Getting the latest news articles from Google news Api
#Write your api key in api_key = ''
import newsapi
from newsapi.newsapi_client import NewsApiClient
newsapi = NewsApiClient(api_key='##################')
top_headlines = newsapi.get_top_headlines(language='en',country='in')

#For specific category news article uncomment this
#entertainment = newsapi.get_top_headlines(category = 'entertainment',language='en',country='in')

url = []
title = []
source = []
for entry in top_headlines['articles']:
    url.append(entry['url'])
    title.append(entry['title'])
    source.append(entry['source']['name'])


#Pre processing the Headlines
stop_words = set(stopwords.words('english'))
def pre_process(text):
    s = '-'
    pos = text.find(s)
    text = text[:pos]
    text = text.replace('[^ws]','')
    text = text.lower()
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    
    text = remove_stopwords(text)
    st = PorterStemmer()
    text = [st.stem(w) for w in text]
    return "".join(text)


#Text_blob for sentiment score
def senti(x):
    return TextBlob(x).sentiment

#Final output as a dictionary where id is Source of Headlines and value is score
#Score is in format:- Sentiment(polarity=0.1, subjectivity=1.0)

dictionary = {}
for i,j in zip(title,source):
    process_text = pre_process(i)
    final_score = senti(process_text)
    dictionary[j] = final_score
print(dictionary)


