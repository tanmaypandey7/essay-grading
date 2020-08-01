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
    data = request.form['essay']
    SERVER_ENDPOINT = 'http://localhost:8501/v1/models/rater1:predict'
    essay = data

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

    score = round(json.loads(r.content)['predictions'][0][0]*10, 1)

    return render_template('index.html', rater1_score='Rater 1 scored {}'.format(score))


@app.route('/rater2', methods=['GET', 'POST'])
def rater2():
    data = request.form['essay']
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

    score = round(json.loads(r.content)['predictions'][0][0]*10, 1)

    return render_template('index.html', rater2_score='Rater 2 scored {}'.format(score))



@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
