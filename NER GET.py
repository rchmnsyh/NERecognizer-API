from flask import Flask, request
# import requests
from nltk.tokenize import word_tokenize
import joblib

crf3 = joblib.load("ner-crf-sgd.sav")

def word2features_input(sent, i):
    word = sent[i]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True

    return features

def prediksi_kalimat(sentence):
    symbols = ['.',',','!','?','&','(',')','-','"']
    sentence = word_tokenize(sentence)
    sentence = [w for w in sentence if not w in symbols]
    
    sentence_feature = [word2features_input(sentence, i) for i in range(len(sentence))]
    prediksi = crf3.predict([sentence_feature])

    return sentence, prediksi

app = Flask(__name__)

@app.route('/<kalimat>', methods=["GET"])
def get(kalimat):
    sentence, prediksi = prediksi_kalimat(kalimat)
    hasil = ""
    for w, pred in zip(sentence, prediksi[0]):
            hasil = hasil + "{} - {}\n".format(w, pred)
    return {
    "chats": [
        {
            "text" : "Kalimat yang dimasukkan: " + kalimat,
            "type" : "text"
        },
        {
            "text" : hasil,
            "type" : "text"
        }
    ]
}

if __name__ == "__main__":
    app.run()