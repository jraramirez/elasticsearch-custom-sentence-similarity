# imports
import re
import pandas as pd
import json
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.utils import simple_preprocess

# function to remove stopwords
stop_words = stopwords.words('english')
def remove_stopwords(texts):
    return [word for word in simple_preprocess(str(texts)) if word not in stop_words]

# function to lemmatize
def lemmatize(words):
    def get_wordnet_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)
    lemmatizer = WordNetLemmatizer()
    wordsLemmatized = []
    return [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words]