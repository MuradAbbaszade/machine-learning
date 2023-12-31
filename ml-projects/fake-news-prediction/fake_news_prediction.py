# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1O05lCcbpKMHQZeMH7d7Qp0wSkInd6Jlf

Importing dependencies
"""

import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

"""Downloading stopwords"""

import nltk
nltk.download("stopwords")
print(stopwords.words("english"))

"""Data processing"""

news_dataset=pd.read_csv('/content/train.csv')
news_dataset.shape
news_dataset.fillna('')
news_dataset['content'] = news_dataset['author']+' '+news_dataset['title']
print(news_dataset["content"])
news_dataset.head()

"""Stemming - is the process reducing word count to root word

Example
actress,actor,acting -> root word is act
"""

port_stem=PorterStemmer()
def stemming(content):
    stemmed_content = re.sub("[^a-zA-Z]",' ',str(content))
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

"""Apply stemming to content"""

news_dataset['content'] = news_dataset['content'].apply(stemming)

print(news_dataset)

"""Split data into label and input"""

X=news_dataset["content"]
Y=news_dataset["label"]

"""Convert text data to numerical data"""

vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X=vectorizer.transform(X)

"""Split input data into train and test data"""

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=1)
print(X_train.shape,Y_train.shape)

"""Training the model"""

model = LogisticRegression()
model.fit(X_train,Y_train)
test_prediction = model.predict(X_test)
test_accuracy_score = accuracy_score(test_prediction,Y_test)
print(test_accuracy_score)