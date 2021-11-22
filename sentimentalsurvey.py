import streamlit as st
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer #3.3
import numpy as np #1.19.5
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from time import sleep

def translate_and_correct(text):
	try:
		trans = GoogleTranslator(source='auto', target='en').translate(text)
	except Exception as e: 
		trans = text
		textBlb = TextBlob(trans)           
		return(str(textBlb.correct()))  

def get_aggregate_sentiment(text, i, outputcsv):
	sid = SentimentIntensityAnalyzer()
	sentiment_cat = 0
	tb_polarity = TextBlob(str(text)).sentiment.polarity
	vader_output = sid.polarity_scores(str(text))
	print(vader_output)
	vader_polarity = vader_output["pos"] - vader_output["neg"]
	sentiment_cont = vader_polarity + tb_polarity
	if vader_polarity > 0:
		sentiment_cat += 1
	if tb_polarity > 0:
		sentiment_cat += 1
	if vader_polarity < 0:
		sentiment_cat -= 1
	if tb_polarity < 0:
		sentiment_cat -= 1
	outputcsv.cont_s[i] = sentiment_cont
	outputcsv.cat_s[i] = sentiment_cat
	outputcsv.vader_sent[i] = vader_polarity
	outputcsv.textblob_sent[i] = tb_polarity
	return(sentiment_cat, sentiment_cont, outputcsv)

def update_statistics(n, n_pos, n_neg, sentiment_cat, sentiment_cont):
	n_pos += (sentiment_cont > 0)		
	n_neg += (sentiment_cont < 0)
	prob_posit_user = (n_pos  / n) 
	prob_negat_user = (n_neg / n) 
	prob_max = np.max([prob_posit_user, prob_negat_user])
	error =  1.96 * np.sqrt((prob_max * (1-prob_max))/n)
	return(prob_posit_user, prob_negat_user, error)

def update_highlights(sentiment_cont, highscores, highscores_i, lowscores, lowscores_i, i):
	if sentiment_cont > np.min(highscores):
		highscores_i = np.array([i, highscores_i[np.argmax(highscores)]])
		highscores = np.array([sentiment_cont, np.max(highscores)])
	elif sentiment_cont < np.max(lowscores):
		lowscores_i = np.array([i, lowscores_i[np.argmin(lowscores)]])
		lowscores = np.array([sentiment_cont, np.min(lowscores)])
	return(highscores, highscores_i, lowscores, lowscores_i)

def plot_current_sentiment_totals(prob_posit_user, prob_negat_user, error, barpl):
	fig, ax = plt.subplots(1, 2)
	ax[0].bar("Positivity", 100*prob_posit_user, yerr=100*error, align='center', alpha=0.5, ecolor='black', capsize=10, color = "#378ce9")
	ax[1].bar("Negativity", 100*prob_negat_user, yerr=100*error, align='center', alpha=0.5, ecolor='black', capsize=10, color = "#378ce9")
	fig.suptitle('Sentiment of texts')
	ax[0].yaxis.grid(True)
	ax[1].yaxis.grid(True)
	plt.tight_layout()
	ax[0].set_ylim([0, 100])
	ax[1].set_ylim([0, 100])
	barpl.pyplot(fig, width = 300)

def emoji_updates(df, pic, posimage, negimage, sentiment_cont):
	if sentiment_cont > 0:
		pic.image(posimage, caption='Text' + str(i+1), width = 100)
		sleep(0.1)
	elif sentiment_cont < 0:
		pic.image(negimage, caption='Text' + str(i+1), width = 100)
		sleep(0.1)

def display_highlights(df, highscores_i, lowscores_i):
	df["Highlights"] = df["text"]
	df = df.loc[np.append(highscores_i, lowscores_i), :]
	df.sort_index(ascending=True, inplace = True)
	df.index.names = ['TextID']	
	st.table(df.loc[:, "Highlights"])








st.markdown("""
<style>
table td:nth-child(1) {
    display: none
}
table th:nth-child(1) {
    display: none
}
</style>
""", unsafe_allow_html=True)

posimage = Image.open('positive_emoji.png')
negimage = Image.open('negative_emoji.png')

uploaded_file = st.file_uploader("Choose a csv file")
if uploaded_file is not None:
	df=pd.read_csv(uploaded_file, header = "infer", sep = ";", index_col = False)

if uploaded_file is not None:
	n = 0
	n_pos = 0
	n_neg = 0

	highscores = np.array([0])
	lowscores = np.array([0])
	highscores_i = np.array([0])
	lowscores_i = np.array([0])
	prog_text = st.empty()
	pic = st.empty()
	barpl = st.empty()
	outputcsv = df.copy()
	outputcsv["cont_s"] = float()
	outputcsv["vader_sent"] = float()
	outputcsv["textblob_sent"] = float()
	outputcsv["cat_s"] = int()

	for i, text in enumerate(df.text):

		#progress indication
		prog_text.text( str(np.round((1+i)/df.shape[0]*100)) + "% done")

		#translation and correction
		prepped_text = translate_and_correct(text)

		#sentiment prediction and storing
		sentiment_cat, sentiment_cont, outputcsv = get_aggregate_sentiment(prepped_text, i, outputcsv)

		#update highlights
		highscores, highscores_i, lowscores, lowscores_i = update_highlights(sentiment_cont, highscores, highscores_i, lowscores, lowscores_i, i)

		#count number obs analyzed
		n += (sentiment_cat != 0)

		# avoid division by zero
		if n == 0:
			continue

		#update statistics
		prob_posit_user, prob_negat_user, error = update_statistics(n, n_pos, n_neg, sentiment_cat, sentiment_cont)

		#sentiment plot update
		plot_current_sentiment_totals(prob_posit_user, prob_negat_user, error, barpl)

		#emoji output update
		emoji_updates(df, pic, posimage, negimage, sentiment_cont)

	#reset printouts
	prog_text.empty()
	pic.empty()
		
	#display final highlights
	display_highlights(df, highscores_i, lowscores_i)

	#write out outputcsv
	print(outputcsv.text[outputcsv["cont_s"] == 0])

	outputcsv.to_csv("outputcsv.csv")