# Library imports
import pandas as pd
import numpy as np
import spacy
import sklearn
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import joblib
import string
from spacy.lang.en.stop_words import STOP_WORDS
from flask import Flask, request, jsonify, render_template
import nltk
from fetch import Fetch
from fetch1 import Fetch1

# Load trained Pipeline
model = joblib.load('sentiment_analysis_model_pipeline.pkl')

# Create the app object
app = Flask(__name__)
fet=Fetch()
fet1=Fetch1()

# creating a function for data cleaning
# from custom_tokenizer_function import CustomTokenizer


# Define predict function
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    url = request.form['user_url']
    print(url)
    all_reviews=fet.collect(url)
    # print(all_reviews)
    # predlist=[]
    # for i in all_reviews:
    #     predictions = model.predict(list(i.split(" ")))[0]
    #     predlist.append(predictions)
    #     print(i)
    #     print(predictions)
    predlist1=fet1.pred(all_reviews)
    percent=round((sum(predlist1)/len(predlist1)),2)*100
    return render_template('index.html', prediction_text=str(percent)+"% positive")



if __name__ == "__main__":
    app.run(debug=True)
