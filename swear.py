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
	
def check_swear(tweet):
	token = nltk.word_tokenize(tweet)
	for swear in swear_list:
		for word in token:
			if word.lower() == swear:
				return True
	return False
	
tweet_list = read_tweet('./full-corpus.csv')
print("total negative: ",tweet_list[1])

negative = 0
caught = 0
for tweet in tweet_list[0]:
	if check_swear(tweet[-1]):
		caught += 1
		if tweet[1]=='negative':
			negative += 1
		# else:
			# print(tweet)
			
print("total caught: ",caught)
			
correctness = negative/caught*100
print("correctness: %.2f %%" %correctness)

completeness = negative/tweet_list[1]*100
print("completeness: %.2f %%" %completeness)