import pickle
import neattext.functions as nfx
import streamlit as st
from keras.models import load_model
from keras_nlp.layers import TokenAndPositionEmbedding, TransformerEncoder
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import lzma

st.set_page_config(page_title='Emotion Detection', layout='centered')
st.title('EMOTION DETECTION')
st.write("Emotions are an integral part of you as an human and I am here to detect those emotions")


@st.experimental_singleton
def loading_model():
    model = load_model('emotions1.h5', custom_objects={'TokenAndPositionEmbedding': TokenAndPositionEmbedding,
                                                       'TransformerEncoder': TransformerEncoder})

    model.load_weights('weights.h5')

    with open('tokenizer.pkl', 'rb') as f:
        token = pickle.load(f)
    return model, token


@st.experimental_singleton
def preprocess_text(the_text):
    lemmatizer = WordNetLemmatizer()
    new_text_ = word_tokenize(the_text)
    lemmatized_text = ' '.join(lemmatizer.lemmatize(w) for w in new_text_)
    removed_stop = nfx.remove_stopwords(lemmatized_text)
    return removed_stop


text = st.text_area('Your text here')

if text:
    new_text = preprocess_text(text)
    model_, tokenizer = loading_model()
    tokenized_text = tokenizer.tokenize(new_text)
    predicted_value = model_.predict(tokenized_text)
    st.write(f'We have detected {predicted_value} in your text')
