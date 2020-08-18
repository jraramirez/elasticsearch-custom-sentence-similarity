import sys
sys.path.append("DataFunctions")
sys.path.append("../DataFunctions")
import defaults
import EDAFunctions as eda
# imports

import re
import pandas as pd
import json
from elasticsearch import Elasticsearch, helpers

from sentence_transformers import SentenceTransformer


def downloadModel():
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    return {}

'''

'''
def getTextVector(data):
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    text = data["inputParagraph"]
    textVector = model.encode(text).tolist() 
    results = {
        "sentence": text, 
        "sentenceVectors": textVector
    }
    return results

def getIndex(indexName):
    es = Elasticsearch(['http://' + defaults.credentials["username"] + ':' + defaults.credentials["password"] + '@' + defaults.credentials["ip_and_port"]], timeout=600)
    doc = {
        "query": {
            "match_all": {}
        }
    }
    result = es.search(index=indexName, body=doc)
    return result

def addTextVectors():
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    data = getIndex("technical-paragraphs")
    data = data["hits"]["hits"]
    actions = [
        {
            "_index": "text-vectors",
            # "_id": d["_source"]["originalParagraph"],
            "_source": {
                "text": d["_source"]["originalParagraph"],
                "vector": model.encode(d["_source"]["originalParagraph"]).tolist() ,
            }
        }
        for d in data
    ]
    print(actions)
    es = Elasticsearch(['http://' + defaults.credentials["username"] + ':' + defaults.credentials["password"] + '@' + defaults.credentials["ip_and_port"]], timeout=600)
    helpers.bulk(es, actions)
    return


def searchText(data):
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    text = data["inputParagraph"]
    textVector = model.encode(text).tolist() 
    es = Elasticsearch(['http://' + defaults.credentials["username"] + ':' + defaults.credentials["password"] + '@' + defaults.credentials["ip_and_port"]], timeout=600)
    doc = {
        "size": 10000,
        "query": {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "cosineSimilarity(params.queryVector, doc['vector'])+1.0",
                    "params": {
                        "queryVector": textVector
                    }
                }
            }
        }
    }
    result = es.search(index="text-vectors", body=doc)
    return result