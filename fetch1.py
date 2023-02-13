import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')


class Fetch1:
    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def pred(self,reviews):
        sentiments=[]
        for review in reviews:
            sentiment = self.sentiment_analyzer.polarity_scores(review)['compound']
            if sentiment >= 0.05:
                sentiments.append(1)
            elif sentiment <= -0.05:
                sentiments.append(0)
            else:
                sentiments.append(0)
        return sentiments
