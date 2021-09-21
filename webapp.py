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

def main():
    '''Review Classifier App with Streamlit'''
    st.title('Neutral Review Classifier ML App')
    st.subheader('NLP and ML App with Streamlit')
    
    activities = ['Prediction', 'NLP']
    
    choice = st.sidebar.selectbox('Choose Activity', activities)
    
    if choice == 'Prediction':
        st.info('Prediction with ML')
        
        review_text = st.text_area('Enter Text', 'Type Here')
        all_ml_models = ['Logistic Regression']
        model_choice = st.selectbox('Choose ML Model', all_ml_models)
        prediction_labels = {'Negative': 0, 'Positive': 1}
        if st.button('Classify'):
            st.text('Original Test ::\n{}'.format(review_text))
        
        
        
    if choice == 'NLP':
        st.info('Natural Language Processing')
    
    
if __name__ == '__main__':
    main()