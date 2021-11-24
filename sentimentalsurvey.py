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
from scipy.stats import ttest_ind

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
	n += 1
	n_pos += (sentiment_cont > 0)		
	n_neg += (sentiment_cont < 0)
	prob_posit_user = (n_pos  / n) 
	prob_negat_user = (n_neg / n) 
	prob_max = np.max([prob_posit_user, prob_negat_user])
	error =  1.96 * np.sqrt((prob_max * (1-prob_max))/n)
	return(n, n_pos, n_neg, prob_posit_user, prob_negat_user, error)

def update_highlights(sentiment_cont, highscores, highscores_i, lowscores, lowscores_i, i):
	if sentiment_cont > np.min(highscores):
		highscores_i = np.array([i, highscores_i[np.argmax(highscores)]])
		highscores = np.array([sentiment_cont, np.max(highscores)])
	elif sentiment_cont < np.max(lowscores):
		lowscores_i = np.array([i, lowscores_i[np.argmin(lowscores)]])
		lowscores = np.array([sentiment_cont, np.min(lowscores)])
	return(highscores, highscores_i, lowscores, lowscores_i)

def plot_current_sentiment_totals(prob_posit_user, prob_negat_user, error, barpl):
	pos_perc = 100*prob_posit_user
	neg_perc = 100*prob_negat_user
	err_perc = 100*error
	ytop_pos = np.min([99 - pos_perc, err_perc])
	ybot_pos = np.min([pos_perc - 1, err_perc])
	ytop_neg = np.min([99 - neg_perc, err_perc])
	ybot_neg = np.min([neg_perc - 1, err_perc])
	fig, ax = plt.subplots(1, 2)
	fig.suptitle('Overall sentiment of texts')
	ax[0].bar("Positivity", pos_perc, align='center', alpha=0.5, ecolor='black', capsize=10, color = "#378ce9") #yerr=100*error,
	ax[1].bar("Negativity", neg_perc, align='center', alpha=0.5, ecolor='black', capsize=10, color = "#378ce9")
	ax[0].errorbar(x = ["Positivity"], y = [pos_perc], yerr = ([ybot_pos], [ytop_pos]))
	ax[1].errorbar(x = ["Negativity"], y = [neg_perc], yerr = ([ybot_neg], [ytop_neg]))
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

def display_highlights(df, highscores_i, lowscores_i, analysis_var):
	st.title("Text samples")
	df["Highlights"] = df[analysis_var]
	df = df.loc[np.append(highscores_i, lowscores_i), :]
	df.sort_index(ascending=True, inplace = True)
	df.index.names = ['TextID']	
	st.table(df.loc[:, "Highlights"])

def display_group_comparison(outputcsv, comparison_var):
	if comparison_var != "No variable selected":
		vals = list(set(outputcsv[comparison_var]))
		if len(vals) != 2:
			st.write("Comparison variable must have exactly two possible values!")
		else:
			group1_label = vals[0]
			group2_label = vals[1]
			group1_ind = outputcsv[comparison_var] == group1_label
			group2_ind = outputcsv[comparison_var] == group2_label
			group1_mean = np.mean(outputcsv.cont_s[group1_ind])
			group2_mean = np.mean(outputcsv.cont_s[group2_ind])
			group1_std = np.std(outputcsv.cont_s[group1_ind])
			group2_std = np.std(outputcsv.cont_s[group2_ind])
			sidedness = np.where(group1_mean > group2_mean, "more positive sentiments", "more negative sentiments")
			d = abs((group1_mean - group2_mean) / np.sqrt((group1_std ** 2 + group2_std **2) / 2))
			d = np.round(d, 3)
			if d < 0.1:
				effsize = "negligible"
			elif d < 0.2:
				effsize = "small"
			elif d < 0.5:
				effsize = "medium"
			else:
				effsize = "large"
			stat, p = ttest_ind(outputcsv.cont_s[group1_ind], outputcsv.cont_s[group2_ind])
			significance = np.where(p < 0.05, "significantly", "not significantly")
			p = np.round(p, 3)
			st.title("Group comparison")
			st.write("Group " + str(group1_label) + " expressed " + str(sidedness) + " than group " + str(group2_label) +
				".  \n The magnitude of this difference can be considered " + effsize + " compared to other findings (Cohen's D: " + str(d) + ").  \n" +
				"The effect is " + str(significance) + " different from zero (p-value: " + str(p) + ").")

################################################################################################################################

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
	form = st.form(key='my_form')
	df=pd.read_csv(uploaded_file, header = "infer", sep = ";", index_col = False)
	analysis_var = form.selectbox("Which column holds the texts?", (["No variable selected"] + list(df.columns)))
	lang = form.selectbox("Select the text language:", ["English", "German"])
	comparison_var = form.selectbox("Add a binary comparison (optional):", (["No variable selected"] + list(df.columns)))
	start = form.form_submit_button(label='Start analyses')
	if start and (analysis_var != "No variable selected"):
		st.title("Sentiment analysis")
		n = 4 #Agresti–Coull correction
		n_pos = 2
		n_neg = 2
		highscores = np.array([0])
		lowscores = np.array([0])
		highscores_i = np.array([0])
		lowscores_i = np.array([0])
		prog_text = st.empty()
		emoji_pic = st.empty()
		barpl = st.empty()
		result_table = st.empty()
		outputcsv = df.copy()
		outputcsv["cont_s"] = float()
		outputcsv["vader_sent"] = float()
		outputcsv["textblob_sent"] = float()
		outputcsv["cat_s"] = int()

		for i, text in enumerate(df[analysis_var]):

			#progress indication
			prog_text.text( str(np.round((1+i)/df.shape[0]*100)) + "% done")

			#translation and correction
			prepped_text = translate_and_correct(text)

			#sentiment prediction and storing
			sentiment_cat, sentiment_cont, outputcsv = get_aggregate_sentiment(prepped_text, i, outputcsv)

			#update highlights
			highscores, highscores_i, lowscores, lowscores_i = update_highlights(sentiment_cont, highscores, highscores_i, lowscores, lowscores_i, i)

			#update statistics
			n, n_pos, n_neg, prob_posit_user, prob_negat_user, error = update_statistics(n, n_pos, n_neg, sentiment_cat, sentiment_cont)
			
			#sentiment plot update
			plot_current_sentiment_totals(prob_posit_user, prob_negat_user, error, barpl)

			#emoji output update
			emoji_updates(df, emoji_pic, posimage, negimage, sentiment_cont)

		#reset printouts
		prog_text.empty()
		emoji_pic.empty()

		#display results table
		result_table.table(pd.DataFrame({"Positive": [str(np.round(prob_posit_user*100, 1)) + "%"], "Negative": [str(np.round(prob_negat_user*100, 1)) + "%"], "CI width": [str(np.round(2*error*100, 1))  + ' points'] }))

		#display final highlights
		display_highlights(df, highscores_i, lowscores_i, analysis_var)

		#display group comparison
		display_group_comparison(outputcsv, comparison_var)

		#write out outputcsv
		#outputcsv.to_csv("outputcsv.csv")