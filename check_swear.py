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
	
	input = full[:]
	print("Total input: %d" % len(input))
	
	return input
	
def check_swear(tweet):
	token = nltk.word_tokenize(tweet)
	for swear in swear_list:
		for word in token:
			if word.lower() == swear:
				return swear
	return ''
	
def check_emo(tweet):
	if not math.isnan(tweet[2]):
		return ['angry',tweet[2]]
	elif not math.isnan(tweet[3]):
		return ['fear',tweet[3]]
	elif not math.isnan(tweet[4]):
		return ['joy',tweet[4]]
	else:
		return ['sadness',tweet[5]]

def count_freq(input):
	output = dict()
	for i in input:
		if i[1] in output:
			output[i[1]][0] += 1
			output[i[1]][1] += i[3]
		else:
			output[i[1]] = [1,i[3]]
	return output

	
tweet_list = read_tweet('./Tweets_crosstab.csv')

output = []
for tweet in tweet_list:
	swear = check_swear(tweet[1])
	if swear != '':
		emotion = check_emo(tweet)
		info = [tweet[0],swear,emotion[0],emotion[1]]
		output.append(info)
			
print("Total caught: ",len(output))

angry,fear,joy,sadness = 0,0,0,0

for i in output:
	if i[2] == 'angry':
		angry += 1
	elif i[2] == 'fear':
		fear += 1
	elif i[2] == 'joy':
		joy += 1
	else:
		sadness += 1

print("Percentage of angry tweets with swear words: %.2f %%" % (angry/len(output)*100))
print("Percentage of fear tweets with swear words: %.2f %%" % (fear/len(output)*100))
print("Percentage of joy tweets with swear words: %.2f %%" % (joy/len(output)*100))
print("Percentage of sadness tweets with swear words: %.2f %%" % (sadness/len(output)*100))

freq = count_freq(output)
for word in freq:
	freq[word] = round(freq[word][1]/freq[word][0],2)
print(freq)