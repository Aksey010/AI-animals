import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import librosa
from flask import Flask, render_template, request
from moduls.sound_prep import __CLASS_LIST__, get_features_full_mean, stack_features_full_mean
import numpy as np
# from autokeras import CUSTOM_OBJECTS
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)
print(app.name)


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/form")
def form():
    return render_template('form.html')


@app.route('/form/result', methods=['POST'])
def sumbit():
    model = load_model('moduls/models/2conv.h5')

    CLASS_LIST = __CLASS_LIST__
    audio = request.files['audio']
    x, sr = librosa.load(audio)
    feat = get_features_full_mean(x, sr)
    data = stack_features_full_mean(feat)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    pred = model.predict(data)
    answer = CLASS_LIST[np.argmax(pred)]

    data = {'answer': answer}

    return render_template('results.html', **data)


@app.route("/contacts")
def contacts():
    return render_template('contacts.html')


if __name__ == '__main__':
    app.run(debug=True)
