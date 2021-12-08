from transformers import pipeline #sentiment
from easynmt import EasyNMT #translation
import re
#problems in translate from french: multiple punctuation marks
#problem in recognizing neutral words: punctuations

ss = ["auf einer bank sitzen SAY Bank SAY hinsetzen"]
# text = re.sub("\!\!+", "!", text) 
# text = re.sub("\?\?+", "?", text) 

translation_analysis = EasyNMT('opus-mt')

for text in ss:
    trans = translation_analysis.translate(text, source_lang = "de", target_lang = "en")
    print(trans)
    sentiment_analysis = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")
    s = sentiment_analysis(trans)[0]
    print(s)

# import string
# import re

# table = str.maketrans({key: None for key in string.punctuation})
# cor = s.translate(table)
# print(cor)