import csv
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import re
import extractor
##import negation
import nltk
#from numba import jit,cuda

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
        def classify(self, bows, prior = False):
                labels = []
                scores = []
                #@jit('float64[::1](float64[:,::1])')
                def log_post(bow):
                        if prior:
                                log_posterior = self.log_prior + np.sum(bow, axis=1)
                        else:
                                log_posterior = np.sum(bow, axis=1)
                        return log_posterior
                #@jit("int64(float64[:])")
                def _l(logP):
                        return np.argmax(logP)
                #@jit("float64(float64[:])")
                def _s(logP):
                        return np.max(logP)
                for bow in tqdm(bows):
##                    if prior:
##                        log_posterior = self.log_prior + np.sum(self.score * bow, axis=1)
##                    else:
##                        log_posterior = np.sum(self.score * bow, axis=1)
                    log_posterior=log_post(np.array(self.score*bow))
                    _label,_score=_l(log_posterior),_s(log_posterior)
                    #labels.append(np.argmax(log_posterior))
                    #scores.append(np.max(log_posterior))
                    labels.append(_label)
                    scores.append(_score)
                return np.asarray(labels), np.asarray(scores)

def read_data():
    pattern = re.compile('\W')
    anger_df = pd.read_csv("result_anger.csv", sep=',', header=None, engine='python', encoding = "utf-8")
    fear_df = pd.read_csv("result_fear.csv", sep=',', header=None, engine='python', encoding = "utf-8")
    joy_df = pd.read_csv("result_joy.csv", sep=',', header=None, engine='python', encoding = "utf-8")
    sadness_df = pd.read_csv("result_sadness.csv", sep=',', header=None, engine='python', encoding = "utf-8")
    # Dataset is now stored in a Pandas Dataframe
    anger_feats = list(anger_df[1])
    fear_feats = list(fear_df[1])
    joy_feats = list(joy_df[1])
    sadness_feats = list(sadness_df[1])

    tweets = anger_feats + fear_feats + joy_feats + sadness_feats
    emotions = [0 for f in anger_feats] + [1 for f in fear_feats] + [2 for f in joy_feats] + [3 for f in sadness_feats]

    return tweets, emotions    

def run(num_samples = 10000, verbose = False, Covid = False, feed_back = None):
        # Extract features

        ext, val_xs, val_ys, count_vectorizer = extractor.run(verbose = verbose, num_samples = num_samples, feed_back = feed_back)
        if Covid:
            val_xs, val_ys = read_data()
##      negationArray = [negation.mark_negation(sent) for sent in val_xs]
        clf = Classifier(ext.score, ext.log_prior, ext.num_classes)
        # Make validation bow
        _, val_bows, sent = extractor._create_bow(val_xs, vectorizer=count_vectorizer, msg_prefix="\n[Validation]")

        # Evaluation
##      val_preds = clf.classify(val_xs, negationArray, count_vectorizer,ext.num_classes)
        val_preds, val_scores = clf.classify(val_bows)
        val_accuracy = accuracy_score(val_ys, val_preds)

        val_cm = confusion_matrix(val_ys, val_preds)        
        print("\n[Validation] Accuracy: {}".format(val_accuracy))
        print("\n[Validation] Confusion matrix: \n{}".format(val_cm))


if __name__ == '__main__':
    run(Covid = True, feed_back = [["johnson", "amazon",  "politics"],[],[],[]])
