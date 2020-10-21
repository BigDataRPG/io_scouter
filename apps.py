import streamlit as st
import numpy as np
import pandas as pd
from pythainlp.tokenize import word_tokenize
import math
import joblib
import base64
import re


st.title("Welcome to IO - Scouter Project ")
st.write("😲 or 🤖 or 🍕  ┬┴┬┴┤ ͜ʖ ͡°) ├┬┴┬┴ Ver 0.0.1 !!")
st.write("This project built for test how much IO you actually have !!")


## LOAD DATA
@st.cache(allow_output_mutation=True)
def load_model():

	model = joblib.load("./model/clf_log_tfidf.joblib")
	vectorizer = joblib.load("./model/vectorizer.joblib")
	# model = joblib.load("C://Users/Boyd/Projects/io_scouter/model/clf_log_tfidf.joblib")
	# vectorizer = joblib.load("C://Users/Boyd/Projects/io_scouter/model/vectorizer.joblib")
	
	return model, vectorizer

loaded_model, vectorizer = load_model()


user_input = st.text_input("Put text here", "...")


def text_treatment(_text):
    
    result = re.sub(r"http\S+", "", _text)
    result = result.replace("\n", "")
    
    return result 


def inference(_user_input):

	treat_text = text_treatment(_user_input)

	test_ls = word_tokenize(treat_text)
	test_ls = " ".join(test_ls)
	X_test = vectorizer.fit_transform([test_ls])

	if X_test.toarray().sum() == 0:
		return 0

	else:
		y_test_prob = loaded_model.predict_proba(X_test)
		y_val = y_test_prob[0][0]
		return y_val

def get_io_score(_y_val):

	if _y_val <= 0.3:
		_y_val = _y_val*1.1
		st.markdown(f"## Your IO Score is {_y_val:.5f} !! ")

	elif _y_val <= 0.35:
		_y_val = _y_val*1.4
		st.markdown(f"## Your IO Score is {_y_val:.5f} !! ")

	elif _y_val <= 0.4:
		_y_val = _y_val*2
		st.markdown(f"## Your IO Score is {_y_val:.5f} !! ")

	elif _y_val > 0.4:
		_y_val = _y_val*9000
		# file_gif = open("C://Users/Boyd/Projects/io_scouter/pict/over9000.gif", "rb")
		file_gif = open("./pict/over9000.gif", "rb")
		contents = file_gif.read()
		data_url = base64.b64encode(contents).decode("utf-8")
		file_gif.close()
		st.markdown(f"## Your IO Score is {_y_val:.5f} !! ")
		st.markdown(
		    f'<center><img src="data:image/gif;base64,{data_url}" width="500">',
		    unsafe_allow_html=True,
		)


y_val = inference(user_input)
get_io_score(y_val)
	














