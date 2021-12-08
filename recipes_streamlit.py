# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 12:29:05 2021

@author: JMada
"""
import pickle as pkl
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import streamlit as st
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest
import pandas as pd
from sentence_transformers import SentenceTransformer
import scipy.spatial
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import torch
nltk.download('stopwords')
nltk.download('punkt')
import heapq
import matplotlib.pyplot as plt
import re
import string
import ast
from unidecode import unidecode
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import config

  

with open("corpus_embeddings.pkl" , "rb") as file4:
  corpus_embeddings=pkl.load(file4)
  
with open("corpus.pkl" , "rb") as file5:
  corpus=pkl.load(file5)
  
with open("recipes.pkl" , "rb") as file6:
  recipes=pkl.load(file6)


embedder = SentenceTransformer('all-MiniLM-L6-v2')


st.title('South Asian Recipe Recommendation System')
st.text('Type in what ingredients you already have')

user_input = st.text_input("Ingredients")

option = st.selectbox(
    'Please enter your preferred diet',
    ('Diabetic Friendly', 'Vegetarian', 'High Protein Vegetarian',
        'Non Vegeterian', 'High Protein Non Vegetarian', 'Eggetarian',
        'Vegan', 'No Onion No Garlic (Sattvic)', 'Gluten Free',
        'Sugar Free Diet'))

st.write('You selected:', option)

if not user_input:
    st.text('Please enter ingredients')
#converting input into string
else:
    query =str(user_input)
    
    query_embeddings = embedder.encode(query,convert_to_tensor=True)
    
    
    # closest_n = 20
    # st.header('Top 20 most similar recipes for your ingredients')
    # for query, query_embedding in zip(query, query_embeddings):
    #     distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]
    
    #     results = zip(range(len(distances)), distances)
    #     results = sorted(results, key=lambda x: x[1])
        
    #     print("\n\n=========================================================")
    #     print("==========================Query==============================")
    #     print("===",query,"=====")
    #     print("=========================================================")
    
    
    #     for idx, distance in results[0:closest_n]:
    #         row_dict = recipes.loc[recipes['all_review']== corpus[idx]]
    #         if row_dict['Diet'].to_string(index=False)==option:
    #             st.write("Score:   ", "(Score: %.4f)" % (1-distance) , "\n")
    #             #print("Score:   ", "(Score: %.4f)" % (1-distance) , "\n" )
    #             st.write("Ingredients Needed:   ", row_dict['Ingredients'].to_string(index=False), "\n")
    #             #print("Ingredients Needed:   ", row_dict['Ingredients'].to_string(index=False), "\n" )
    #             st.write("Name of Dish:  " , row_dict['RecipeName'].to_string(index=False) , "\n")
    #             #print("Name of Dish:  " , row_dict['RecipeName'].to_string(index=False) , "\n")
    #             st.write("Diet:  " , row_dict['Diet'].to_string(index=False) , "\n")
    #             #print("Diet:  " , row_dict['Diet'].to_string(index=False) , "\n")
    #             st.write("Link to Recipe:  " , row_dict['URL'].to_string(index=False) , "\n")
    #             #print("Link to Recipe:  " , row_dict['URL'].to_string(index=False) , "\n")
    #             st.write("-------------------------------------------")
    #             #print("-------------------------------------------")
            
    top_k = min(5, len(corpus))
    cos_scores = util.pytorch_cos_sim(query_embeddings, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)
      
    st.header('Top 5 most similar recipes for your ingredients')  
      
    for score, idx in zip(top_results[0], top_results[1]):
        row_dict = recipes.loc[recipes['all_review']== corpus[idx]]
        if row_dict['Diet'].to_string(index=False)==option:
            st.write('(Score: {:.4f})'.format(score))
            st.write("Ingredients Needed:   ", row_dict['Ingredients'].to_string(index=False), "\n")
            #print("Ingredients Needed:   ", row_dict['Ingredients'].to_string(index=False), "\n" )
            st.write("Name of Dish:  " , row_dict['RecipeName'].to_string(index=False) , "\n")
            #print("Name of Dish:  " , row_dict['RecipeName'].to_string(index=False) , "\n")
            st.write("Diet:  " , row_dict['Diet'].to_string(index=False) , "\n")
            #print("Diet:  " , row_dict['Diet'].to_string(index=False) , "\n")
            st.write("Link to Recipe:  " , row_dict['URL'].to_string(index=False) , "\n")
            #print("Link to Recipe:  " , row_dict['URL'].to_string(index=False) , "\n")
            st.write("-------------------------------------------")
            #print("-------------------------------------------")



 