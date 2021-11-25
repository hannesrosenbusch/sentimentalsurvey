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
from easynmt import EasyNMT
import base64

def translate_and_correct(text, model, lang, outputcsv, i):
	if text == "-":
		trans = "-"
	elif lang == "German":
		trans = model.translate(text, source_lang = "de", target_lang = "en")
	elif lang == "English":
		textBlb = TextBlob(text)           
		trans = str(textBlb.correct())
	else:
		trans = model.translate(text,  target_lang = "en")
	outputcsv.trans[i] = trans
	return(trans, outputcsv)

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
	outputcsv.sentiment_continuous[i] = np.round(sentiment_cont * 100)
	outputcsv.sentiment_categorical[i] = sentiment_cat
	outputcsv.vader_sent[i] = np.round(vader_polarity * 100)
	outputcsv.textblob_sent[i] = np.round(tb_polarity * 100)
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

def update_highlights(sentiment_cont, highscores, highscores_i, lowscores, lowscores_i, i, df):
	if (sentiment_cont > np.min(highscores)) and (df.loc[i, "text_low"] not in df.loc[highscores_i, "text_low"].values):
		print(i)
		print(df.loc[i, analysis_var])
		print(df.loc[highscores_i, analysis_var])
		print(df.loc[i, analysis_var] not in df.loc[highscores_i, analysis_var].values)
		highscores_i = np.array([i, highscores_i[np.argmax(highscores)]])
		highscores = np.array([sentiment_cont, np.max(highscores)])
	elif sentiment_cont < np.max(lowscores) and (df.loc[i, "text_low"] not in df.loc[lowscores_i, "text_low"]):
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
	prog_text.text( str(np.round((1+i)/df.shape[0]*100)) + "% done")
	if sentiment_cont > 0:		
		pic.image(posimage, caption='Text' + str(i+1), width = 100)
	elif sentiment_cont < 0:
		pic.image(negimage, caption='Text' + str(i+1), width = 100)

def display_highlights(df, highscores_i, lowscores_i, analysis_var):
	st.title("Text samples")
	df["Highlights"] = df[analysis_var]
	df = df.loc[np.append(highscores_i, lowscores_i), :]
	st.table(df.loc[:, "Highlights"])

def display_group_comparison(outputcsv, comparison_var, df):
	if comparison_var != "No variable selected":
		outputcsv[comparison_var] = df[comparison_var]
		vals = list(set(outputcsv[comparison_var]))
		if len(vals) != 2:
			st.write("Comparison variable must have exactly two possible values!")
		else:
			group1_label = vals[0]
			group2_label = vals[1]
			group1_ind = outputcsv[comparison_var] == group1_label
			group2_ind = outputcsv[comparison_var] == group2_label
			group1_mean = np.mean(outputcsv.sentiment_continuous[group1_ind])
			group2_mean = np.mean(outputcsv.sentiment_continuous[group2_ind])
			group1_std = np.std(outputcsv.sentiment_continuous[group1_ind])
			group2_std = np.std(outputcsv.sentiment_continuous[group2_ind])
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
			stat, p = ttest_ind(outputcsv.sentiment_continuous[group1_ind], outputcsv.sentiment_continuous[group2_ind])
			significance = np.where(p < 0.05, "significantly", "not significantly")
			p = np.round(p, 3)
			st.title("Group comparison")
			st.write("Group " + str(group1_label) + " expressed " + str(sidedness) + " than group " + str(group2_label) +
					".  \n The magnitude of this difference can be considered " + effsize + " (Cohen's D: " + str(d) + ").  \n" +
					"The effect is " + str(significance) + " different from zero (p-value: " + str(p) + ").")
		return(outputcsv)

def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="outputcsv.csv" >Download outputs as csv</a>'
    return(href)


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

st.markdown(
    """ <style>
            div[role="radiogroup"] >  :first-child{
                display: none !important;
            }
        </style>
        """,
    unsafe_allow_html=True
)


posimage = Image.open('positive_emoji.png')
negimage = Image.open('negative_emoji.png')
uploaded_file = st.file_uploader("Choose a csv file")
model = EasyNMT('opus-mt')
if uploaded_file is not None:
	form = st.form(key='my_form')
	df=pd.read_csv(uploaded_file, header = "infer", sep = ";", index_col = False, encoding = "latin1")
	analysis_var = form.selectbox("Which column holds the texts?", (["No variable selected"] + list(df.columns)))
	lang = form.selectbox("Select the text language:", ["English", "German", "Other"])
	comparison_var = form.selectbox("Add a binary comparison (optional):", (["No variable selected"] + list(df.columns)))
	start = form.form_submit_button(label='Start analyses')
	
	if start and (analysis_var != "No variable selected"):
		st.title("Sentiment analysis")
		n = 4 #Agresti–Coull correction
		n_pos = n_neg = 2
		highscores = lowscores = highscores_i = lowscores_i = np.array([0])
		prog_text, emoji_pic, barpl, result_table = st.empty(), st.empty(), st.empty(), st.empty(),
		df["text_low"] = [text.lower() for text in df[analysis_var]]
		outputcsv = df.iloc[:, [0,1,2]].copy()
		outputcsv["text"] = df[analysis_var]
		outputcsv["trans"] = str()
		outputcsv["sentiment_categorical"] = int()
		outputcsv["sentiment_continuous"] = outputcsv["vader_sent"]  = outputcsv["textblob_sent"] = float()

		for i, text in enumerate(df[analysis_var]):

			#translation and correction
			prepped_text, outputcsv = translate_and_correct(text, model, lang, outputcsv, i)

			#sentiment prediction and storing
			sentiment_cat, sentiment_cont, outputcsv = get_aggregate_sentiment(prepped_text, i, outputcsv)
			if sentiment_cat != 0: #for efficiency

				#update highlights
				highscores, highscores_i, lowscores, lowscores_i = update_highlights(sentiment_cont, highscores, highscores_i, lowscores, lowscores_i, i, df)

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
		outputcsv = display_group_comparison(outputcsv, comparison_var, df)

		#download outputcsv
		st.title("Download")
		st.markdown(get_table_download_link(outputcsv), unsafe_allow_html=True)