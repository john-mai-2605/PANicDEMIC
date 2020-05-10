import csv
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import pandas as pd
import re
import extractor
import negation
import nltk
import intensity

class Classifier():
	def __init__(self, score, log_prior, num_classes = 4):
		self.score = score
		self.log_prior = log_prior
		self.num_classes = num_classes
	def classify(self, val_xs, negationArray, intensityArray, vectorizer, num_classes):
		labels=[]
		for i,sent in enumerate(val_xs):
			sent = [w.lower() for w in nltk.word_tokenize(sent)]
			word2id = [vectorizer.vocabulary_.get(word, 'UNK') for word in sent]
			scoreList = []
			for numClass in range(num_classes):
				score_feat = [self.score[numClass][id_] if id_ != 'UNK' else 0 for id_ in word2id] 
				scoreList.append(score_feat)
			final_score = [np.sum(np.multiply(score_feat, np.multiply(negationArray[i],intensityArray[i]))) for score_feat in scoreList]
			if(i%700==3):
				print(sent,end="\n    Score Array : ")
				print(scoreList,end="\nIntensity array : ")
				print(intensityArray[i],end="\n Negation array : ")
				print(negationArray[i],end="\n    Fianl score : ")
				print(final_score)
			labels.append(np.argmax(final_score))
		return np.asarray(labels)
def run(num_samples = 10000, verbose = True):
	# Extract features
	ext, val_xs, val_ys, count_vectorizer = extractor.run(verbose=False)	
	negationArray = [negation.mark_negation(sent) for sent in val_xs]
	intensityScoreArray = [intensity.intensityScores(sent) for sent in val_xs]
	clf = Classifier(ext.score, ext.log_prior, ext.num_classes)
	# Make validation bow
	_, val_bows = extractor._create_bow(val_xs, vectorizer=count_vectorizer, msg_prefix="\n[Validation]")
	# Evaluation
	val_preds = clf.classify(val_xs, negationArray, intensityScoreArray, count_vectorizer,ext.num_classes)
	val_accuracy = accuracy_score(val_ys, val_preds)
	if verbose:
		print("\n[Validation] Accuracy: {}".format(val_accuracy))

if __name__ == '__main__':
    run()
