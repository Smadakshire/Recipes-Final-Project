# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 17:11:48 2021

@author: JMada
"""

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest
import pandas as pd
from sentence_transformers import SentenceTransformer
import scipy.spatial
import pickle as pkl
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import torch
import pickle as pkl
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import heapq
import matplotlib.pyplot as plt
import streamlit as st
import string
import ast
from unidecode import unidecode
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from collections import Counter
import config

#load the recipes dataset
recipes=pd.read_csv('indian_food_data.csv')
#drop unused columns
recipes=recipes.drop(columns=['Srno','RecipeName', 'Ingredients', 'Course', 'Cuisine', 'PrepTimeInMins','CookTimeInMins','TotalTimeInMins', 'Servings', 'Instructions', 'TranslatedInstructions'])
#rename columns
recipes.rename(columns={'TranslatedRecipeName':'RecipeName','TranslatedIngredients':'Ingredients'},inplace=True) #change column names
recipes['RecipeName'].drop_duplicates()
#drop rows w/o ingredients
recipes = recipes.dropna(subset=['Ingredients'])

#load new dataset that removed recipes with non-alphabetic ingredients
recipes=pd.read_csv('cleaned_recipes.csv')
recipes=recipes.drop(columns=['Unnamed: 0'])

#recipes['all_review'] = recipes['Ingredients'].apply(lambda x: re.split('[,.]', x)) #splits using comma delimiter to create list of ingredients
recipes['all_review'] = recipes['Ingredients'].apply(lambda x: re.sub(r'(?<=[.,])(?=[^\s])', r' ', x)) #add space after commas
recipes['all_review'] = recipes['all_review'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x)) #remove punctuation and other stuff
recipes["all_review"] = recipes['all_review'].str.replace('\d+', '') #removes number digits
recipes['all_review'] = recipes['all_review'].apply(lambda x: re.sub(' +',' ',x)) #removes double spaces

def lower_case(input_str):
    input_str = input_str.lower()
    return input_str

recipes['all_review'] = recipes['all_review'].apply(lambda x: lower_case(x)) # makes text lower case

recipes["all_review"] = recipes['all_review'].apply(lambda x: x[1:].split(' ')) #splits ingredients into list of strings

#recipes["all_review"] = recipes['all_review'].str.replace('[^\w\s]','')

#remove stopwords from list of strings
stop= stopwords.words('english')
recipes["all_review"] = recipes['all_review'].apply(lambda x: [item for item in x if item not in stop])

measures = [
        "teaspoon",
        "t",
        "tsp.",
        "tablespoon",
        "tablespoons",
        "T",
        "tbl.",
        "tb",
        "tbsp",
        "tsps"
        "tbsps"
        "fluid ounce",
        "fl oz",
        "gill",
        "cup",
        "c",
        "pint",
        "p",
        "pt",
        "fl pt",
        "quart",
        "q",
        "qt",
        "fl qt",
        "gallon",
        "g",
        "gal",
        "ml",
        "milliliter",
        "millilitre",
        "cc",
        "mL",
        "l",
        "liter",
        "litre",
        "L",
        "dl",
        "deciliter",
        "decilitre",
        "dL",
        "bulb",
        "level",
        "heaped",
        "rounded",
        "whole",
        "pinch",
        "medium",
        "slice",
        "pound"
        "pounds",
        "lb",
        "#",
        "ounce",
        "oz",
        "mg",
        "milligram",
        "milligramme",
        "g",
        "gram",
        "gramme",
        "kg",
        "kilogram",
        "kilogramme",
        "x",
        "of",
        "mm",
        "millimetre",
        "millimeter",
        "cm",
        "centimeter",
        "centimetre",
        "m",
        "meter",
        "metre",
        "inch",
        "in",
        "milli",
        "centi",
        "deci",
        "hecto",
        "kilo",
        "tsp",
        "cups",
        "ounces",
        "grams",
        "milligrams",
        "milliliters",
        "pinches",
        "gallons",
        "teaspoons",
        "inches",
        "kilograms",
        "lbs",
        ]

words_to_remove = [
        "lukewarm",
        "breasts",
        "boiled",
        "per",
        "homemade",
        "fresh",
        "minced",
        "chopped",
        "a",
        "red",
        "bunch",
        "and",
        "clove",
        "cloves",
        "or",
        "large",
        "extra",
        "sprig",
        "ground",
        "handful",
        "free",
        "small",
        "pepper",
        "virgin",
        "range",
        "from",
        "dried",
        "sustainable",
        "black",
        "peeled",
        "use",
        "pieces",
        "piece",
        "higher",
        "seed",
        "for",
        "finely",
        "freshly",
        "sea",
        "quality",
        "white",
        "ripe",
        "few",
        "source",
        "organic",
        "flat",
        "smoked",
        "ginger",
        "sliced",
        "green",
        "picked",
        "the",
        "stick",
        "plain",
        "plus",
        "mixed",
        "mint",
        "bay",
        "your",
        "optional",
        "fennel",
        "serve",
        "unsalted",
        "baby",
        "fat",
        "ask",
        "natural",
        "skin",
        "roughly",
        "into",
        "such",
        "cut",
        "good",
        "brown",
        "grated",
        "trimmed",
        "powder",
        "yellow",
        "dusting",
        "knob",
        "frozen",
        "homemade"
        "on",
        "deseeded",
        "low",
        "runny",
        "balsamic",
        "cooked",
        "streaky",
        "sage",
        "rasher",
        "zest",
        "pin",
        "groundnut",
        "breadcrumb",
        "halved",
        "grating",
        "stalk",
        "light",
        "tinned",
        "dry",
        "soft",
        "rocket",
        "bone",
        "colour",
        "washed",
        "skinless",
        "leftover",
        "splash",
        "removed",
        "dijon",
        "thick",
        "big",
        "hot",
        "drained",
        "sized",
        "chestnut",
        "watercress",
        "fishmonger",
        "english",
        "raw",
        "flake",
        "tbsp",
        "leg",
        "pine",
        "wild",
        "if",
        "fine",
        "herb",
        "shoulder",
        "cube",
        "dressing",
        "with",
        "chunk",
        "spice",
        "thumb",
        "garam",
        "new",
        "little",
        "punnet",
        "peppercorn",
        "shelled",
        "other",
        "chopped",
        "salt",
        "taste",
        "can",
        "sauce",
        "water",
        "diced",
        "package",
        "italian",
        "shredded",
        "divided",
        "all",
        "purpose",
        "crushed",
        "juice",
        "more",
        "bell",
        "needed",
        "thinly",
        "boneless",
        "half",
        "cubed",
        "jar",
        "seasoning",
        "extract",
        "sweet",
        "baking",
        "beaten",
        "heavy",
        "seeded",
        "tin",
        "uncooked",
        "crumb",
        "style",
        "thin",
        "nut",
        "coarsely",
        "spring",
        "strip",
        "rinsed",
        "cherry",
        "root",
        "quartered",
        "head",
        "softened",
        "container",
        "crumbled",
        "frying",
        "lean",
        "cooking",
        "roasted",
        "warm",
        "whipping",
        "thawed",
        "corn",
        "pitted",
        "sun",
        "kosher",
        "bite",
        "toasted",
        "lasagna",
        "split",
        "melted",
        "degree",
        "lengthwise",
        "packed",
        "pod",
        "anchovy",
        "rom",
        "prepared",
        "juiced",
        "fluid",
        "floret",
        "room",
        "active",
        "seasoned",
        "mix",
        "deveined",
        "lightly",
        "anise",
        "thai",
        "size",
        "unsweetened",
        "torn",
        "wedge",
        "sour",
        "marinara",
        "dark",
        "temperature",
        "garnish",
        "bouillon",
        "loaf",
        "shell",
        "reggiano",
        "canola",
        "parmigiano",
        "round",
        "canned",
        "crust",
        "long",
        "broken",
        "ketchup",
        "bulk",
        "cleaned",
        "condensed",
        "sherry",
        "provolone",
        "cold",
        "soda",
        "cottage",
        "spray",
        "pecorino",
        "shortening",
        "part",
        "bottle",
        "sodium",
        "cocoa",
        "grain",
        "french",
        "roast",
        "stem",
        "link",
        "firm",
        "mild",
        "dash",
        "boiling",
        "oil",
        "chopped",
        "vegetable oil",
        "chopped oil",
        "skin off",
        "bone out",
        "according",
        "required"]

recipes["all_review"] = recipes['all_review'].apply(lambda x: [item for item in x if item not in measures])
recipes["all_review"] = recipes['all_review'].apply(lambda x: [item for item in x if item not in words_to_remove])




#join the list of strings back together
recipes["all_review"] = recipes['all_review'].str.join(" ")

recipes_sentences = recipes.set_index("all_review")
recipes_sentences = recipes_sentences["RecipeName"].to_dict()
recipes_sentences_list = list(recipes_sentences.keys())
len(recipes_sentences_list)

list(recipes_sentences.keys())[:5]

recipes_sentences_list = [str(d) for d in tqdm(recipes_sentences_list)]

corpus = recipes_sentences_list
embedder = SentenceTransformer('all-MiniLM-L6-v2')
corpus_embeddings = embedder.encode(corpus,convert_to_tensor=True)

with open("corpus_embeddings.pkl" , "wb") as file4:
  pkl.dump(corpus_embeddings,file4)
  
with open("corpus.pkl" , "wb") as file5:
  pkl.dump(corpus,file5)
  
with open("recipes.pkl" , "wb") as file6:
  pkl.dump(recipes,file6)

