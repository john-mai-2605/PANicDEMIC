import csv
import nltk
import random
import pandas as pd
import math
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer as lem

swear_list = ['fuck','fucks','fucked','shit','damn','suck','sucks','wtf','wth','stupid','dick','heck']

def read_tweet(filename):
	
	reader = pd.read_csv(filename, sep='\t', header=None, engine='python', skiprows=2, encoding = "utf-16")
	
	full = []
	id = list(reader[0])
	text = list(reader[1])
	angry = list(reader[2])
	fear = list(reader[3])
	joy = list(reader[4])
	sadness = list(reader[5])
	
	for i in range(len(id)):
		full.append([id[i],text[i],angry[i],fear[i],joy[i],sadness[i]])
	
	random.shuffle(full)
	
	input = full[:3000]
	print("total input: %d" % len(input))
	
	angry_total = 0
	for row in input:
		if not math.isnan(row[2]):
			angry_total += 1
	
	return [input,angry_total]
	
def check_swear(tweet):
	token = nltk.word_tokenize(tweet)
	for swear in swear_list:
		for word in token:
			if word.lower() == swear:
				return swear
	return ''
	
def check_emo(tweet):
	if tweet[2] != '':
		return ['angry',float(tweet[2])]
	elif tweet[2] != '':
		return ['fear',float(tweet[2])]
	elif tweet[2] != '':
		return ['joy',float(tweet[2])]
	else:
		return ['sadness',float(tweet[2])]

tweet_list = read_tweet('./Tweets_crosstab.csv')

output = []
for tweet in tweet_list[0]:
	swear = check_swear(tweet[1])
	if swear != '':
		emotion = check_emo(tweet)
		info = [tweet[0],swear,emotion[0],emotion[1]]
		output.append(info)
			
print("total angry: ",tweet_list[1])
print("total caught: ",len(output))

angry = 0
for i in output:
	if i[2] == 'angry':
		angry += 1

print("percentage of angry tweets with swear words: %.2f %%" % (angry/len(output)*100))

print("percentage of tweets with swear words in all angry comments: %.2f %%" % (len(output)/tweet_list[1]*100))