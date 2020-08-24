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


def connectES(credentials):
    es = Elasticsearch(['http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"]], timeout=600)
    return es


def downloadModel():
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    return {}


def createTextVectorsIndex(indexName):
    doc = {
        "mappings": {
            "properties": {
                "vector": {
                    "type": "dense_vector",
                    "dims": 768
                },
                "text" : {
                    "type" : "keyword"
                }
            }
        }
    }
    es = connectES(defaults.credentials)
    es.indices.create(indexName, body=doc)
    return True


def getTextVector(data):
    """
    Get vector of a text data using the trained model encoder
    :return: the vector by trained model encoder of the text data
    """
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    text = data["inputParagraph"]
    textVector = model.encode(text).tolist() 
    results = {
        "text": text, 
        "textVectors": textVector
    }
    return results


def getIndex(indexName):
    es = connectES(defaults.credentials)
    doc = {
        'size' : 10000,
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
            "_source": {
                "text": d["_source"]["originalParagraph"],
                "vector": model.encode(d["_source"]["originalParagraph"]).tolist() ,
            }
        }
        for d in data
    ]
    es = connectES(defaults.credentials)
    helpers.bulk(es, actions)
    return


def searchText(data):
    """
    Search for similar vectors using the vector of a text data
    :return: the list of data corresponding to the similar vectors found
    """
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    text = data["inputParagraph"]
    limit = data["limit"]
    textVector = model.encode(text).tolist() 
    es = connectES(defaults.credentials)
    doc = {
        "size": limit,
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