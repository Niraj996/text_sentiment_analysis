#For internal Python codec registry
import codecs

#translate hindi to english
from deep_translator import GoogleTranslator

#for analysis the sentiment of text
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


""" Read file data 'codecs' lib provides access to the internal python codec registry
    i.e convert text to bytes."""
with codecs.open('/home/niraj/Documents/pythonVenv/pract_venv/simple_file.txt', encoding='utf-8') as f:
    sentences = f.readlines()

#Translate sentences read into English so that VADER lib can process translated rext for sentiment analysis.
#They have these polarity_scores(), that have "compound" score based on that evaluation is done.
"""positive sentiment: compound score >= 0.05
Neutral sentiment : compound score > -0.05 and compound score < 0.05
Negative sentiment : compound score <= -0.05 """

for sentence in sentences:
    translated_text = GoogleTranslator(source='auto', target='en').translate(sentence)
    print(translated_text)
    analyzer = SentimentIntensityAnalyzer()
    sentiment_dict = analyzer.polarity_scores(translated_text)

    print("\nTranslated Sentence=", translated_text,"\nDictionary= ", sentiment_dict)
    if sentiment_dict['compound']>=0.05:
        print("It is a Positive Sentence")
    elif sentiment_dict['compound'] <= -0.05:
        print("It is a Nagative Sentence")
    else:
        print("It is a Neutral Sentence")