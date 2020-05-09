import csv
import nltk
import random
import pandas as pd
import math
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer as lem

score = {'shit': 0.55, 'wtf': 0.58, 'damn': 0.56, 'fucked': 0.68, 'fuck': 0.61, 'sucks': 0.65, 'stupid': 0.62, 'suck': 0.51, 'dick': 0.48, 'heck': 0.68}


def read_tweet(filename):
	
	reader = pd.read_csv(filename, sep='\t', header=None, engine='python', skiprows=2, encoding = "utf-16")
	
	id = list(reader[0])
	text = list(reader[1])
	
	input = []
	
	for i in range(len(id)):
		input.append([id[i],text[i]])
	
	return input
	
def swear_score(tweet,score):
	token = nltk.word_tokenize(tweet)
	
	mark = 0
	
	for swear in score:
		for word in token:
			if word.lower() == swear:
				if word.upper() == word:
					mark += 2*score[swear]
				else:
					mark += score[swear]
				
	return mark

id_tweet = read_tweet('./Tweets_crosstab.csv')

output = []
for i in id_tweet:
	tweet_score = swear_score(i[1],score)
	output.append([i[0],i[1],tweet_score])
	
write = pd.DataFrame(output, columns=['Id', 'Tweet','Swear Score'])
write.to_csv('swear_score.csv')
	
	