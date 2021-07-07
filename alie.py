from flask import Flask, render_template,request
import pandas as pd
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np
import json
import random

import sys

data_file = open('resps.json').read()
resps = json.loads(data_file)

lemmatizer=WordNetLemmatizer()

app = Flask(__name__)

import pickle

model=pickle.load(open('alie.pkl','rb'))
bot=load_model('Alie_bot.h5')


@app.route("/home")
def home():
    return render_template("home.html",title='Home')

@app.route("/session")
def session():
    return render_template("session.html",title='New Session')

@app.route("/chat")
def chat():
    return render_template("chat.html",title='Chat with Alie')


@app.route("/predict",methods=['POST'])
def predict():
    q1  = request.form.get("q1")
    q2  = request.form.get("q2")
    q3  = request.form.get("q3")
    q4  = request.form.get("q4")
    q5  = request.form.get("q5")
    q6  = request.form.get("q6")
    q7  = request.form.get("q7")
    text=""
    text+=q1+"."+q2+"."+q3+"."+q4+"."+q5+"."+q6+"."+q7
    output= model.predict([text])
    out=""
    for ele in output:
        out+=ele
    resl = get_res(out)
    return render_template("dummy2.html",text=out,title='Result of Analysis',result=resl)

def get_res(out):
    for r in resps['responses']:
        if r['tag']==out:
            result=random.choice(r['response'])
            return result

@app.route("/chatbot")
def chatbot():
    chat_msg=request.args.get('msg')
    app.logger.info('testing info log',chat_msg)
    input_text=[chat_msg]
    df=pd.read_csv('response.csv')
    tokenizer_t = joblib.load('tokenizer_t.pkl')
    vocab = joblib.load('vocab.pkl')
    df_input = pd.DataFrame(input_text,columns=['questions'])
    doc_without_stopwords = []
    entry = df_input['questions'][0]
    tokens = tokenizer(entry)
    doc_without_stopwords.append(' '.join(tokens))
    df_input['questions'] = doc_without_stopwords
    t = tokenizer_t
    entry = entry = [df_input['questions'][0]]
    encoded = t.texts_to_sequences(entry)
    encoded_ip = pad_sequences(encoded, maxlen=16, padding='post')
    pred = np.argmax(bot.predict(encoded_ip))
    words = df_input.questions[0].split()
    if len([w for w in words if w in vocab])==0 :
        pred = 1
    upper_bound = df.groupby('labels').get_group(pred).shape[0]
    r = np.random.randint(0,upper_bound)
    responses = list(df.groupby('labels').get_group(pred).responses)
    response=responses[r]
    return str(response)

def tokenizer(entry):
    tokens = entry.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [re_punc.sub('', w) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [lemmatizer.lemmatize(w.lower()) for w in tokens]
    tokens = [word.lower() for word in tokens if len(word) > 1]
    return tokens




