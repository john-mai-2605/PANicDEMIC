import csv
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import pandas as pd
import re

import extractor

class Classifier():
	def __init__(self, score, log_prior, num_classes = 4):
		self.score = score
		self.log_prior = log_prior
		self.num_classes = num_classes
	def classify(self, bows):
		labels = []
		for bow in tqdm(bows):
			log_posterior = self.log_prior + np.sum(self.score * bow, axis=1)
			labels.append(np.argmax(log_posterior))
		return np.asarray(labels)
def run(num_samples = 10000, verbose = True):
	# Extract features
	ext, val_xs, val_ys, count_vectorizer = extractor.run()
	clf = Classifier(ext.score, ext.log_prior, ext.num_classes)
	# Make validation bow
	_, val_bows = extractor._create_bow(val_xs, vectorizer=count_vectorizer, msg_prefix="\n[Validation]")
	# Evaluation
	val_preds = clf.classify(val_bows)
	val_accuracy = accuracy_score(val_ys, val_preds)
	if verbose:
		print("\n[Validation] Accuracy: {}".format(val_accuracy))

if __name__ == '__main__':
    run()
