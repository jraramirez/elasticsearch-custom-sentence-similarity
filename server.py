from flask import Flask, request
from flask_restful import Resource, Api

import DataFunctions.Functions as df

app = Flask(__name__)

@app.route('/init-vectors', methods=['GET', 'POST']) 
def initVectors():
    if df.createTextVectorsIndex("text-vectors"):
        df.addTextVectors()
    return {}

@app.route('/text-vector', methods=['POST']) 
def textVector():
    if request.method == 'POST':
        data = request.json
        result = df.getTextVector(data)
        return result
    return {}

@app.route('/text-match', methods=['GET', 'POST']) 
def textMatch():
    if request.method == 'POST':
        data = request.json
        result = df.searchText(data)
        return result
    return {}

if __name__ == '__main__':
     app.run(debug=True, port='5002', host='0.0.0.0')