# -*- coding: utf-8 -*-
# from crypt import methods
from flask import Flask, render_template, jsonify, request
# import json
import requests
# from functions import *
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
from nltk.corpus import stopwords
import contractions
import joblib

app = Flask(__name__)

# METEO_API_KEY = "4e07a22d9905acf30e155e5628a0fc74"

# if METEO_API_KEY is None:
#     # URL de test :
#     METEO_API_URL = "https://samples.openweathermap.org/data/2.5/forecast?lat=0&lon=0&appid=xxx"
# else: 
#     # URL avec clé :
#     METEO_API_URL = "https://api.openweathermap.org/data/2.5/forecast?lat=48.883587&lon=2.333779&appid=" + METEO_API_KEY

# @app.route("/<a>/<b>")
# def hello(a, b):
#     c = str(int(a) + int(b))
#     return c

@app.route("/predict/", methods=['GET', 'POST'])
def hello_world():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template("tag-gen.html", href = 'Type a question')
    else:
        text = request.form['question'] + ' ' + request.form['titre']
        model, sw, lb, vec = import_model()
        tags = treat_text_get_tags(text, model, lb, vec, sw)
        return render_template("tag-gen.html", href = ' '.join(tags[0]))

def cleaner(text):
    """Remove Html tags, extra spaces, ; put in lowercase"""

    text = BeautifulSoup(text, 'html.parser')
    text = text.get_text(strip=True)
    text = contractions.fix(text) # remove contractions 's => is...
    text = re.sub(r"\n", " ", text) # match all literal Line Feed (New line) pattern then replace them by a single whitespace
    text = re.sub(r'\s+', ' ', text) # match all one or more whitespace then replace them by a single whitespace
    text = text.lower()

    return text

def tokeniser(text):
    return nltk.RegexpTokenizer(r'[a-zA-Z]{2,}|c#?|.net').tokenize(text)

def remove_stopwords(list_of_words, sw):
    """remove common words in english by using nltk.corpus's list"""

    list_of_words = [w for w in list_of_words if not w in sw]
    return list_of_words

def lem_text(list_of_words):
    """Lemmatization of the text"""

    lemmatizer = nltk.WordNetLemmatizer()
    list_of_words = [lemmatizer.lemmatize(w) for w in list_of_words] # Lemmatize each words
    return list_of_words

def text_treatment(input_txt, sw):
   txt = cleaner(input_txt)
   txt = tokeniser(txt)
   txt = remove_stopwords(txt, sw)
   txt = lem_text(txt)
   # vectorizer needs a  list of strings
   # it then tokenise it
   txt = [' '.join(txt)] 
   return txt

def vectorize(x, vec):
   return vec.transform(x)

def get_best_tags(clf, X, lb, n_tags=3):
    decfun = clf.decision_function(X)
    best_tags = np.argsort(decfun)[:, :-(n_tags+1): -1]
    return lb.classes_[best_tags]

def treat_text_get_tags(x, model, lb, vec, sw):
    x_clean = text_treatment(str(x), sw)
    x_vec = vectorize(x_clean, vec)
    return get_best_tags(model, x_vec, lb)

def import_model():
    model = joblib.load("models\clf_svc.pkl")
    sw = joblib.load("models\sw.pkl")
    lb = joblib.load("models\multilabel.pkl")
    vec = joblib.load("vec_tfidf.joblib")
    return (model, sw, lb, vec)

# @app.route('/dashboard/')
# def dashboard():
#     return render_template("dashboard.html")

# @app.route('/api/meteo/')
# def meteo():
#     response = requests.get(METEO_API_URL)
#     content = json.loads(response.content.decode('utf-8'))
    
#     if response.status_code != 200:
#         return jsonify({
#             'status': 'error',
#             'message': 'La requête à l\'API météo n\'a pas fonctionné. Voici le message renvoyé par l\'API : {}'.format(content['message'])
#         }), 500

#     data = [] # On initialise une liste vide
#     for prev in content["list"]:
#         datetime = prev['dt'] * 1000
#         temperature = prev['main']['temp'] - 273.15 # Conversion de Kelvin en °c
#         temperature = round(temperature, 2)
#         data.append([datetime, temperature])

#     return jsonify({
#       'status': 'ok', 
#       'data': data
#     })

# NEWS_API_KEY = '9ba03827ace1429886410c47567efb8c'

# if NEWS_API_KEY is None:
#     # URL de test :
#     NEWS_API_URL = "https://s3-eu-west-1.amazonaws.com/course.oc-static.com/courses/4525361/top-headlines.json" # exemple de JSON
# else:
#     # URL avec clé :
#     NEWS_API_URL = "https://newsapi.org/v2/top-headlines?sortBy=publishedAt&pageSize=100&language=fr&apiKey=" + NEWS_API_KEY

# @app.route('/api/news/')
# def get_news():
 
#     response = requests.get(NEWS_API_URL)

#     content = json.loads(response.content.decode('utf-8'))

#     if response.status_code != 200:
#         return jsonify({
#             'status': 'error',
#             'message': 'La requête à l\'API des articles d\'actualité n\'a pas fonctionné. Voici le message renvoyé par l\'API : {}'.format(content['message'])
#         }), 500

#     keywords, articles = extract_keywords(content["articles"])

#     return jsonify({
#         'status'   : 'ok',
#         'data'     :{
#             'keywords' : keywords[:100], # On retourne uniquement les 100 premiers mots
#             'articles' : articles
#         }
#     })

if __name__ == "__main__":
    app.run(debug=True)

# from app.src.main import Predict

# @app.route('/predict/<path:path_X_test>')
# def predict(path_X_test):
#     """
#     Example: 
#     path_X_test=app/data/dataset_single.csv
#     Link: 
#         http://127.0.0.1:5000/predict/app/data/dataset_single.csv
#     """
#     isinstance(path_X_test, str)
#     y_pred = Predict(path_X_test)
#     return {'status': 'OK', 'y_pred': y_pred.tolist()}


# if _name_ == '_main_':
#     app.run('0.0.0.0', 5000)

# postman pour tester les API