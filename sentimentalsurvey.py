import streamlit as st
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
import string

@st.experimental_memo()
def initiate_global_vars():
	neutral_words = list(pd.read_csv("neutral_words.csv", header = "infer", sep = ",", index_col = False, encoding = "latin1")["words"].astype("str"))
	neutral_words.append("")
	posimage = Image.open('positive_emoji.png')
	negimage = Image.open('negative_emoji.png')
	translation_analysis = EasyNMT('opus-mt')
	sentiment_analysis = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")
	punct_table = str.maketrans({key: None for key in string.punctuation})
	return(neutral_words, posimage, negimage, translation_analysis, sentiment_analysis, punct_table)

def translate_and_correct(translation_analysis, lang, outputcsv):
	outputcsv["text"] = outputcsv["text"].str.replace("\!\!+", "!") 
	outputcsv["text"] = outputcsv["text"].str.replace("\?\?+", "?") 
	if lang == "German":
		trans = translation_analysis.translate(outputcsv["text"], source_lang = "de", target_lang = "en")
	elif lang == "Spanish":
		trans = translation_analysis.translate(outputcsv["text"], source_lang = "es", target_lang = "en")
	elif lang == "French":
		trans = translation_analysis.translate(outputcsv["text"], source_lang = "fr", target_lang = "en")
	elif lang == "English":
		textBlb = TextBlob(outputcsv["text"])           
		trans = str(textBlb.correct())
	else:
		try:
			trans = translation_analysis.translate(outputcsv["text"],  target_lang = "en")
		except:
			stop("translation error")
	
	trans = [trans[i] if outputcsv.text[i] != "-" else "-" for i in range(len(trans))]
	outputcsv["trans"] = trans
	return(outputcsv)

def get_aggregate_sentiment(sentiment_analysis, neutral_words, outputcsv, i, punct_table):
	print(outputcsv.trans[i].lower().translate(punct_table))
	if outputcsv.trans[i].lower().translate(punct_table) in neutral_words:
		sentiment_cont = sentiment_cat = 0
	else:
		try:
			s = sentiment_analysis(outputcsv.trans[i])[0]

			if s['label'] == "NEGATIVE":
				sign = -1
			else:
				sign = 1
			sentiment_cont = sign * s['score']
			sentiment_cat = sign * (s['score'] > 0.95)
		except:
			sentiment_cont = sentiment_cat = 0
			print("error at:" + outputcsv.trans[i])
	outputcsv.sentiment_continuous[i] = np.round(sentiment_cont * 100)
	outputcsv.sentiment_categorical[i] = sentiment_cat
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
		highscores_i = np.array([i, highscores_i[np.argmax(highscores)]])
		highscores = np.array([sentiment_cont, np.max(highscores)])
	elif sentiment_cont < np.max(lowscores) and (df.loc[i, "text_low"] not in df.loc[lowscores_i, "text_low"]):
		lowscores_i = np.array([i, lowscores_i[np.argmin(lowscores)]])
		lowscores = np.array([sentiment_cont, np.min(lowscores)])
	return(highscores, highscores_i, lowscores, lowscores_i)

def plot_current_sentiment_totals(prob_posit_user, prob_negat_user, error, barpl):
	plt.close('all')
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

def emoji_updates(df, pic, posimage, negimage, sentiment_cont, i):
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

#load materials
neutral_words, posimage, negimage, translation_analysis, sentiment_analysis, punct_table = initiate_global_vars()
print(neutral_words)
uploaded_file = st.file_uploader("Choose a csv file")

if uploaded_file is not None:
	form = st.form(key='my_form')
	df=pd.read_csv(uploaded_file, header = "infer", sep = ";", index_col = False, encoding = "latin1")
	analysis_var = form.selectbox("Which column holds the texts?", (["No variable selected"] + list(df.columns)))
	lang = form.selectbox("Select the text language:", ["English", "French", "German", "Spanish", "Other"])
	comparison_var = form.selectbox("Add a binary comparison (optional):", (["No variable selected"] + list(df.columns)))
	start = form.form_submit_button(label='Start analyses')
	if start and (analysis_var != "No variable selected"):
		start = time.time()

		with st.spinner('Analyzing contents...'):
			n = 4 #Agresti???Coull correction
			n_pos = n_neg = 2
			highscores = lowscores = highscores_i = lowscores_i = np.array([0])
			prog_text, emoji_pic, barpl, result_table = st.empty(), st.empty(), st.empty(), st.empty()
			df[analysis_var] =  df[analysis_var].astype("str")
			df["text_low"] = [text.lower() for text in df[analysis_var]]
			df["text_low"] =  df["text_low"].astype("str")
			outputcsv = df.iloc[:, [0,1]].copy()
			outputcsv["text"] = df[analysis_var]
			outputcsv["trans"] = str()
			outputcsv["sentiment_categorical"] = int()
			outputcsv["sentiment_continuous"] = float()

			#translation and correction
			outputcsv = translate_and_correct(translation_analysis, lang, outputcsv)
		
		st.title("Sentiment analysis")

		for i, text in enumerate(outputcsv.trans):
			#sentiment prediction and storing
			sentiment_cat, sentiment_cont, outputcsv = get_aggregate_sentiment(sentiment_analysis, neutral_words, outputcsv, i, punct_table)

			if sentiment_cat != 0: #for efficiency and mutual informativeness

				#update highlights
				highscores, highscores_i, lowscores, lowscores_i = update_highlights(sentiment_cont, highscores, highscores_i, lowscores, lowscores_i, i, df)

				#update statistics
				n, n_pos, n_neg, prob_posit_user, prob_negat_user, error = update_statistics(n, n_pos, n_neg, sentiment_cat, sentiment_cont)
				
				#sentiment plot update
				#plot_current_sentiment_totals(prob_posit_user, prob_negat_user, error, barpl)

				#emoji output update
				#emoji_updates(df, emoji_pic, posimage, negimage, sentiment_cont, i)

		#sentiment plot update
		plot_current_sentiment_totals(prob_posit_user, prob_negat_user, error, barpl)

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
		print(time.time() - start)
		import winsound
		duration = 1000  # milliseconds
		freq = 440  # Hz
		winsound.Beep(freq, duration)