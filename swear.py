import csv
import nltk
import random
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer as lem

swear_list = ['fuck','fucks','fucked','shit','damn','suck','sucks','wtf','wth','stupid','dick','heck']

def read_tweet(filename):
	
	f = open(filename, encoding="utf8")
	reader = csv.reader(f)
	next(reader)
	next(reader)
	
	angry_total = 0
	list = []
	for row in reader:
		list.append(row)
		if row[2] != '':
			angry_total += 1
		if len(list)==200:
			break
			
	random.shuffle(list)
	return [list,angry_total]
	
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

tweet_list = read_tweet('./tweet_data.csv')

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
