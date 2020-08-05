from flask import Flask
from flask import request, jsonify
from flask import render_template

import pickle
import json
import requests

import sys
sys.path.append("./src")
import utils
import config

import numpy as np

app = Flask(__name__)


@app.route('/rater1', methods=['GET', 'POST'])
def rater1():
    data = request.get_json()
    SERVER_ENDPOINT = 'http://localhost:8501/v1/models/rater1:predict'
    essay = data['essay']

    with open('input/tokenizer_essays.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    preprocessed_text = utils.preprocess(["dummy", essay],
                                         tokenizer,
                                         config.MAX_LEN)[1]

    preprocessed_text = np.reshape(preprocessed_text, (1, 300))

    payload = {
        "instances": preprocessed_text.tolist()
    }

    r = requests.post(
        SERVER_ENDPOINT,
        json=payload)

    return jsonify({'result': json.loads(r.content)}), 201


@app.route('/rater2', methods=['GET', 'POST'])
def rater2():
    data = request.get_json()
    SERVER_ENDPOINT = 'http://localhost:8501/v1/models/rater2:predict'
    essay = data['essay']

    with open('input/tokenizer_essays.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    preprocessed_text = utils.preprocess(["dummy", essay],
                                         tokenizer,
                                         config.MAX_LEN)[1]

    preprocessed_text = np.reshape(preprocessed_text, (1, 300))

    payload = {
        "instances": preprocessed_text.tolist()
    }

    r = requests.post(
        SERVER_ENDPOINT,
        json=payload)

    return jsonify({'result': json.loads(r.content)}), 201

@app.route('/rater', methods=['GET', 'POST'])
def rater():
    data = request.form['essay']
    SERVER_ENDPOINT1 = 'http://0.0.0.0:8080/rater1'
    r1 = requests.post(
        SERVER_ENDPOINT1,
        json={'essay': data})
    score1 = json.loads(r1.text)['result']['predictions'][0][0]

    SERVER_ENDPOINT2 = 'http://0.0.0.0:8080/rater2'
    r2 = requests.post(
        SERVER_ENDPOINT2,
        json={'essay': data})
    score2 = json.loads(r2.text)['result']['predictions'][0][0]

    return render_template('index.html', rater_score=f'Rater 1 scored {round(score1*10, 1)} and Rater 2 scored {round(score2*10, 1)} ')


@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
