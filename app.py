# -*- coding: utf-8 -*-
# from crypt import methods
from flask import Flask, render_template, jsonify, request
# import json
# import requests
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
    model = joblib.load("clf_svc.pkl")
    sw = joblib.load("sw.pkl")
    lb = joblib.load("multilabel.pkl")
    vec = joblib.load("vec_tfidf.pkl")
    return (model, sw, lb, vec)

# if __name__ == '__main__':
#     app.debug = True
#     app.run(port=8080)

if __name__ == "__main__":
    app.run(debug=True)
    