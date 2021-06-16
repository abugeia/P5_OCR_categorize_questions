# -*- coding: utf-8 -*-
import re
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
import pickle

def extract_keywords(texts):
    """
    Prends une liste de textes et en détecte les mots les plus fréquents. Cette fonction renvoie 
    la liste des textes et le comptage des mots.
    """

    keywords = {}
    articles = []

    for i, text in enumerate(texts):
        # Extraction des éléments selon la structure JSON renvoyée par l'API NEWSAPI.ORG
        source = text["source"]["name"]
        title = text["title"]
        description = text["description"]
        url = text["url"]
        content = text["content"]

        # Stockage des articles dans la variable articles
        articles.append({'title': title, 'url': url, 'source':source})

        # Détection des mots clés (mots les plus utilisés)
        text = str(title) + ' ' + str(description) + ' ' + str(content)
        words = normalise_and_get_words(text)

        # Comptage des mots
        for w in words :
            if w not in keywords:
                keywords[w] = {'cnt': 1, 'articles':[i]}
            else:
                keywords[w]['cnt'] += 1
                if i not in keywords[w]['articles']:
                    keywords[w]['articles'].append(i)

    # Tri des mots, du plus utilisé au moins utilisé
    keywords = [{'word':word, **data} for word,data in keywords.items()] 
    keywords = sorted(keywords, key=lambda x: -x['cnt'])

    return keywords, articles

def load_stop_words():
    """
    Charge la liste des stopwords français (les mots très utilisés qui ne sont pas porteurs de sens comme LA, LE, ET, etc.)
    """

    words = []
    # Ouverture du fichier "stop_words.txt"
    with open("stop_words.txt") as f:
        for word in f.readlines():
            words.append(word[:-1])
    return words

def normalise_and_get_words(text):
    """
    Prends un texte, le formate puis renvoie tous les mots significatifs qui le constituent
    """

    stop_words = load_stop_words()

    # Utilisation des expressions régulières (voir https://docs.python.org/3.7/library/re.html et https://openclassrooms.com/fr/courses/4425111-perfectionnez-vous-en-python/4464009-utilisez-des-expressions-regulieres)
    text = re.sub("\W"," ",text) # suppression de tous les caractères autres que des mots
    text = re.sub(" \d+", " ", text) # suppression des nombres
    text = text.lower() # convertit le texte en minuscules
    words = re.split("\s",text) # sépare tous les mots du texte

    words = [w for w in words if len(w) > 2] # suppression des mots de moins de 2 caractères
    words = [w for w in words if w not in stop_words] # suppression des stopwords
    return words

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

def text_treatment(input_txt):
   txt = cleaner(input_txt)
   txt = tokeniser(txt)
   txt = remove_stopwords(txt)
   txt = lem_text(txt)
   # vectorizer needs a  list of strings
   # it then tokenise it
   txt = [' '.join(txt)] 
   return txt

def vectorize(x):
   return vectorizer_tfidf.transform(x)

def get_best_tags(clf, X, lb, n_tags=3):
    decfun = clf.decision_function(X)
    best_tags = np.argsort(decfun)[:, :-(n_tags+1): -1]
    return lb.classes_[best_tags]

def import_model():
    model = joblib.load("models\clf_svc.pkl")
    sw = joblib.load("models\sw.pkl")
    lb = joblib.load("models\multilabel.pkl")

    # vectorizer_tfidf = pickle.load(open("models\tfidf.pickle", "rb"))