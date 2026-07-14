import asyncio
import sys

# Force Windows to use the Selector event loop instead of Proactor (fixes socket errors)
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model  # <-- Added missing import
import streamlit as st

##Load the word index (Crucial for preprocessing and decoding!)
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

##Load the model cleanly
@st.cache_resource  # Caches the model so it doesn't reload and freeze on every click!
def load_my_model():
    return load_model('Simple_RNN_IMDB.h5')

try:
    model = load_model('Simple_RNN_IMDB.h5')
    st.write("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

model = load_my_model()

## Helper Functions

## To Decode the Review
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

## Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


## Streamlit app layout
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as Positive or Negative')

## User Input
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        # Preprocess the text
        preprocessed_input = preprocess_text(user_input)
        
        # Run prediction (Fixed: Missing model.predict call)
        prediction = model.predict(preprocessed_input)
        score = prediction[0][0]
        
        # Determine sentiment
        sentiment = 'Positive' if score > 0.5 else 'Negative'
        
        ## Display the results
        st.subheader("Result:")
        if sentiment == 'Positive':
            st.success(f'Sentiment: **{sentiment}**')
        else:
            st.error(f'Sentiment: **{sentiment}**')
            
        st.info(f'Prediction Score: **{score:.4f}**')
else: 
    st.write('Please enter a movie review.')