import streamlit as st 
import joblib,os
import spacy
import pandas as pd
nlp = spacy.load('en_core_web_sm')
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# load Vectorizer For Gender Prediction
news_vectorizer = open("pickle/tfidf_vectorizer","rb")
news_cv = joblib.load(news_vectorizer)

def load_prediction_models(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model

# Get the Keys
def get_key(val,my_dict):
	for key,value in my_dict.items():
		if val == value:
			return key

def main():
	"""Review Classifier"""
	st.title("Review Classifier")
	# st.subheader("ML App with Streamlit")
	html_temp = """
	<div style="background-color:blue;padding:10px">
	<h1 style="color:white;text-align:center;">Streamlit ML App </h1>
	</div>
	"""
	st.markdown(html_temp,unsafe_allow_html=True)

	activity = ['Prediction','NLP']
	choice = st.sidebar.selectbox("Select Activity",activity)


	if choice == 'Prediction':
		st.info("Prediction with ML")

		news_text = st.text_area("Enter News Here","Type Here")
		all_ml_models = ["LR","NB"]
		model_choice = st.selectbox("Select Model",all_ml_models)

		prediction_labels = {'Negative': 0,'Positive': 1}
		if st.button("Classify"):
			st.text("Original Text::\n{}".format(news_text))
			vect_text = news_cv.transform([news_text]).toarray()
			if model_choice == 'LR':
				predictor = load_prediction_models("pickle/lr_classifier_tuned")
				prediction = predictor.predict(vect_text)
				# st.write(prediction)
			elif model_choice == 'NB':
				predictor = load_prediction_models("pickle/nb_classifier")
				prediction = predictor.predict(vect_text)
				# st.write(prediction)
			final_result = get_key(prediction,prediction_labels)
			st.success("Reviewgorized as:: {}".format(final_result))

	if choice == 'NLP':
		st.info("Natural Language Processing of Text")
		raw_text = st.text_area("Enter News Here","Type Here")
		nlp_task = ["Tokenization","Lemmatization","NER","POS Tags"]
		task_choice = st.selectbox("Choose NLP Task",nlp_task)
		if st.button("Analyze"):
			st.info("Original Text::\n{}".format(raw_text))

