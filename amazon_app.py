import numpy as np
import pandas as pd
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import streamlit as st
from PIL import Image

# Load the dataset
amazon_df = pd.read_csv('amazon_product.csv')

# Remove unnecessary itrems
amazon_df.drop('id',axis=1,inplace=True)


nltk.download('punkt_tab')
# Define tokenizer and stemmer
stemmer = SnowballStemmer('english')
def tokenize_stem(text):
    tokens = nltk.word_tokenize(text.lower()) #seperates words as single unit
    stemmed = [stemmer.stem(w) for w in tokens] #converts different forms of a word to single
    return " ".join(stemmed)
#this function first converts the word into lowercase, then word tokenize it, then removes the extra forms of the word and return best form(eg. love for lover, loved)

# Create stemmed tokens column
amazon_df['stemmed_tokens'] = amazon_df.apply(lambda row:tokenize_stem(row['Title']+" "+row['Description']),axis=1)

# Define TF-IDF vectorizer and cosine similarity function
tfidv = TfidfVectorizer(tokenizer = tokenize_stem)
def cosine_sim(txt1,txt2):
    matrix = tfidv.fit_transform([txt1,txt2])
    return cosine_similarity(matrix)[0][1]

# Define search function
def search_product(query):
    stemmed_query = tokenize_stem(query)
    #calculating cosine similarity between query and stemmed token columns
    amazon_df['similarity'] = amazon_df['stemmed_tokens'].apply(lambda x:cosine_sim(stemmed_query,x))
    result = amazon_df.sort_values(by=['similarity'],ascending=False).head(10)[['Title','Description','Category']]
    return result

# WEB APP

# Centering with simple HTML
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)

img = Image.open('amazon_logo.png')
st.image(img,width=400)

st.markdown("<h1 style='text-align: center;'>Search Engine and Product Recommendation System on Amazon Data</h1>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

query = st.text_input("Enter Product Name")
submit = st.button('Search')
if submit:
    results = search_product(query)
    st.write(results)