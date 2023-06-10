import joblib
import pandas as pd
import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
import sys
import os

this_dir = os.path.dirname(__file__) # Path to loader.py
sys.path.append(os.path.join(this_dir, 'rf_model.pkl'))


def predictSentiment_RNN_RF(text):
    rnn_model = load_model('rnn_model.h5')
    rf_model = joblib.load('rf_model.pkl')
    input_text = np.array([text])

    
    with open('tokenizer.json') as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)

    input_sequence = tokenizer.texts_to_sequences(input_text)
    input_padded = pad_sequences(input_sequence, maxlen=100)

    rnn_features = rnn_model.predict(input_padded)

    prediction = rf_model.predict(rnn_features)

    # Map numerical labels to class labels
    class_labels = ["negative", "neutral", "positive"]
    predicted_labels = [class_labels[label] for label in prediction]

    return predicted_labels[0]

st.title("Tweet Sentiment Prediction")
st.text("Predict the sentiment of your Tweet using a hybrid RNN-RF model")

text = st.text_input("Enter the text you'd like to analyze:")

if st.button("Predict"):
    prediction = predictSentiment_RNN_RF(text)
    st.write("The sentiment of the text is", prediction)

pd.show_versions()