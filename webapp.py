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


# Vectorizer
review_vectorizer = open('pickle/tfidf_vectorizer', 'rb')
review_cv = joblib.load(review_vectorizer)

# Load Models
def load_prediction_models(model_file):
    loaded_models = joblib.load(open(os.path.join(model_file), 'rb'))
    return loaded_models

def main():
    '''Review Classifier App with Streamlit'''
    st.title('Neutral Review Classifier ML App')
    st.subheader('NLP and ML App with Streamlit')
    
    activities = ['Prediction', 'NLP']
    
    choice = st.sidebar.selectbox('Choose Activity', activities)
    
    if choice == 'Prediction':
        st.info('Prediction with ML')
        
        review_text = st.text_area('Enter Text', 'Type Here')
        all_ml_models = ['Logistic Regression', 'NB']
        model_choice = st.selectbox('Choose ML Model', all_ml_models)
        prediction_labels = {'Negative': 0, 'Positive': 1}
        if st.button('Classify'):
            st.text('Original Test ::\n{}'.format(review_text))
            vect_text = review_cv.transform([review_text]).toarray()
            if model_choice == 'LR': 
                predictor = load_prediction_models('pickle/lr_classifier_tuned.pkl')
                prediction = predictor.predict(vect_text)
                st.write(prediction)
        
        # 19:05 video timestamp
        
        
    if choice == 'NLP':
        st.info('Natural Language Processing')
    
    
if __name__ == '__main__':
    main()