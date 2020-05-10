import csv
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import pandas as pd
import re
import extractor
##import negation
import nltk

class Classifier():
        def __init__(self, score, log_prior, num_classes = 4):
                self.score = score
                self.log_prior = log_prior
                self.num_classes = num_classes
##      def classify(self, val_xs, negationArray, vectorizer, num_classes):
##              labels=[]
##              for i,sent in enumerate(val_xs):
##                      sent = [w.lower() for w in nltk.word_tokenize(sent)]
##                      word2id = [vectorizer.vocabulary_.get(word, 'UNK') for word in sent]
##                      scoreList = []
##                      for numClass in range(num_classes):
##                              score_feat = [self.score[numClass][id_] if id_ != 'UNK' else 0 for id_ in word2id] 
##                              scoreList.append(score_feat)
##                      final_score = [np.sum(np.dot(score_feat, negationArray[i])) for score_feat in scoreList]
##                      labels.append(np.argmax(final_score))
        def classify(self, bows):
                labels = []
                for bow in tqdm(bows):
                        log_posterior = self.log_prior + np.sum(self.score * bow, axis=1)
                        labels.append(np.argmax(log_posterior))
                return np.asarray(labels)       
def run(num_samples = 10000, verbose = False):
        # Extract features
        ext, val_xs, val_ys, count_vectorizer, data = extractor.run() 
##      negationArray = [negation.mark_negation(sent) for sent in val_xs]
        clf = Classifier(ext.score, ext.log_prior, ext.num_classes)
        # Make validation bow
        val_bows_list = []
        for i in data:
                _, val_bow_data = extractor._create_bow(random.sample(i, 50000), vectorizer=count_vectorizer, msg_prefix="\n[Validation]")
                val_bows_list.append(val_bow_data)
##        rand1 = random.sample(data1_list, 50000)
##        rand2 = random.sample(data2_list, 50000)
        _, val_bows = extractor._create_bow(val_xs, vectorizer=count_vectorizer, msg_prefix="\n[Validation]")
##        _, val_bows_data1 = extractor._create_bow(rand1, vectorizer=count_vectorizer, msg_prefix="\n[Validation]")
##        _, val_bows_data2 = extractor._create_bow(rand2, vectorizer=count_vectorizer, msg_prefix="\n[Validation]")

        # Evaluation
##      val_preds = clf.classify(val_xs, negationArray, count_vectorizer,ext.num_classes)
        val_data_list = map(lambda x: clf.classify(x), val_bows_list)
##        val_data1 = clf.classify(val_bows_data1)
##        val_data2 = clf.classify(val_bows_data2)
        result_data= []
        for i in val_data_list:
                fd = nltk.FreqDist(i)
                result_data.append(list(fd.items()))
##        print("____________GET RESULT FOR DATA1")
##        fd1 = nltk.FreqDist(val_data1)
##        print(list(fd1.items()))
##        print("____________GET RESULT FOR DATA2")
##        fd2 = nltk.FreqDist(val_data2)
##        print(list(fd2.items()))
        print(result_data)
        val_preds = clf.classify(val_bows)
        val_accuracy = accuracy_score(val_ys, val_preds)
        if verbose:
                print("\n[Validation] Accuracy: {}".format(val_accuracy))

if __name__ == '__main__':
    run()
