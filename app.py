from flask import Flask, request, jsonify, render_template
from pickle import load
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

app = Flask(__name__)

def preprocess(raw_tweet):
    letters_only = re.sub("[^a-zA-Z]", " ",raw_tweet)
    letters_only = letters_only.lower()
    words = letters_only.split()
    words = [w for w in words if not w in stopwords.words("english")]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    clean_sent = " ".join(words)
    return clean_sent

def predict(tweet):
    vectorizer = load(open('pickle/countvectorizer.pkl', 'rb'))
    classifier = load(open('pickle/logit_model.pkl', 'rb'))
    clean_tweet = preprocess(tweet)
    clean_tweet_encoded = vectorizer.transform([clean_tweet])
    prediction = classifier.predict(clean_tweet_encoded)
    return prediction

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    tweet = request.form['tweet']
    prediction = predict(tweet)
    if prediction == 0:
        sentiment = "Negative"
        return render_template('negative.html', tweet=tweet, sentiment=sentiment)
    elif prediction == 1:
        sentiment = "Positive"
        return render_template('positive.html', tweet=tweet, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)

