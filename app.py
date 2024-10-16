# REQUIRED LIBRARIES

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import string



## READING DATA

data_fake = pd.read_csv(r"C:\Users\Samiksha\Desktop\Fake news detection\dataset\Fake.csv")
data_true = pd.read_csv(r"C:\Users\Samiksha\Desktop\Fake news detection\dataset\True.csv")


## DATA CLEANING & PREPROCESSING

data_fake['target'] ='fake'
data_true['target'] = 'true'

data_fake_manual_testing = data_fake.tail(10)
for i in range(23480, 23470, -1):
    data_fake.drop([i], axis = 0, inplace = True)

data_true_manual_testing = data_true.tail(10)
for i in range(21416, 21406, -1):
    data_fake.drop([i], axis = 0, inplace = True)
    
data_fake_manual_testing["target"] = 'fake' 
data_true_manual_testing["target"] = 'true'


# concatenate dataframes
data = pd.concat([data_fake,data_true]).reset_index (drop = True)

# shuffle the data
from sklearn.utils import shuffle
data = shuffle(data)
data = data.reset_index(drop = True)


# removing the date
data.drop(["date"],axis=1,inplace=True)

# removing the title
data.drop(["title"],axis=1,inplace=True)

def wordopt(text):
    text = text.lower()
    text = re.sub('/[.*?/]', ' ', text)
    text = re.sub("/W", " ", text)
    text = re.sub('https?:/S+', ' ', text)
    text = re.sub('<.*?>+', ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('/n', ' ', text)
    text = re.sub('/w*/d/w*', ' ', text)
    return text

data['text'] = data['text'].apply( wordopt)



## SPLIT DATA 

x = data["text"]
y = data["target"]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)



## MODEL FITTING


# Logistic Regression

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(xv_train, y_train)


# Decision Tree

from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)


# Gradient Boosting

from sklearn.ensemble import GradientBoostingClassifier

GB = GradientBoostingClassifier(random_state = 0)
GB.fit(xv_train, y_train)


# Random Forest

from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(random_state = 0)
RF.fit(xv_train, y_train)


# Function to convert model output to human-readable label
def output_label(n):
    if n == 'fake':
        return "Fake News"
    elif n == 'true':
        return "True News"

# Function for manual testing
def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GB = GB.predict(new_xv_test)
    pred_RF = RF.predict(new_xv_test)
    
    print("/n/nLR Prediction: {} /nDT Prediction: {} /nGB Prediction: {} /nRF Prediction: {}".format(
        output_label(pred_LR[0]),
        output_label(pred_DT[0]),
        output_label(pred_GB[0]),
        output_label(pred_RF[0]),
        
    ))




# WEBSITE

st.title('Fake News Detector')
input_text = st.text_input("Enter News Article")

# Function for model prediction
def prediction(input_text):
    input_data = vectorization.transform([input_text])
    prediction = LR.predict(input_data)
    prediction = DT.predict(input_data)
    prediction = GB.predict(input_data)
    prediction = RF.predict(input_data)
    return prediction[0]

# Display prediction result on the web app
if st.button("Check News"):
    if input_text:
        pred = prediction(input_text)
        result_label = output_label(pred)
        st.write(f'The News is {result_label}')

