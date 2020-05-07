import csv
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer as lem

swear_list = ['fuck','fucks','fucked','fucking','shit','damn','suck','sucks','wtf','wth','stupid','dick','heck']

def read_tweet(filename):
	
	negative = 0
	
	f = open(filename, encoding="utf8")
	reader = csv.reader(f)
	
	list = []
	for row in reader:
		list.append(row)
		if row[1] == 'negative':
			negative += 1
		if len(list)==3000:
			break
	
	return [list, negative]
	
	
def swear_score(swear):
	positive = 0
	neutral = 0
	negative = 0
	caught = 0
	for tweet in tweet_list[0]:
		token = nltk.word_tokenize(tweet[-1])
		for word in token:
			if word.lower() == swear:
				caught += 1
				if tweet[1]=='negative':
					negative += 1
				elif tweet[1]=='neutral':
					neutral += 1
				elif tweet[1]=='negative':
					positive += 1
	print(swear,"total: %d,negative: %d, positive: %d, neutral: %d"%(caught, negative, positive, neutral))
	
	
	
tweet_list = read_tweet('./full-corpus.csv')
for swear in swear_list:
	score = swear_score(swear)