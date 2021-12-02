import streamlit as st
import tokenizers
import numpy as np 
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import time
from scipy.stats import ttest_ind
from easynmt import EasyNMT #translation
from transformers import pipeline #sentiment
import base64

def initiate_global_vars():
	neutral_words = list(pd.read_csv("neutral_words.csv", header = "infer", sep = ",", index_col = False, encoding = "latin1")["words"])
	posimage = Image.open('positive_emoji.png')
	negimage = Image.open('negative_emoji.png')
	translation_analysis = EasyNMT('opus-mt')
	sentiment_analysis = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")
	return(neutral_words, posimage, negimage, translation_analysis, sentiment_analysis)

neutral_words, posimage, negimage, translation_analysis, sentiment_analysis = initiate_global_vars()
df=pd.read_csv("sentimenttest.csv", header = "infer", sep = ";", index_col = False, encoding = "latin1")
lang = "German"
i = 1
df["trans"] = ""


def translate_and_correct(text, translation_analysis, lang, outputcsv, i):
	if lang == "German":
		trans = translation_analysis.translate(text, source_lang = "de", target_lang = "en")
	elif lang == "English":
		textBlb = TextBlob(text)           
		trans = str(textBlb.correct())
	else:
		trans = translation_analysis.translate(text,  target_lang = "en")
	trans = [trans[i] if outputcsv.text[i] != "-" else "-" for i in range(len(trans))]
	outputcsv["trans"] = trans
	return(trans, outputcsv)

trans, out = translate_and_correct(df["text"], translation_analysis, lang, df, i)

print(trans)

