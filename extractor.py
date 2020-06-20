import csv
import random
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import pandas as pd
import re
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
def get_data():
    pattern = re.compile('\W')
    df = pd.read_csv("Tweets_crosstab.csv", sep='\t', header=None, engine='python', skiprows=2, encoding = "utf-16")
    # Dataset is now stored in a Pandas Dataframe
    anger_feats = list(df[1][df[2].notnull()])
    fear_feats = list(df[1][df[3].notnull()])
    joy_feats = list(df[1][df[4].notnull()])
    sadness_feats = list(df[1][df[5].notnull()])

    tweets = anger_feats + fear_feats + joy_feats + sadness_feats
    emotions = [0 for f in anger_feats] + [1 for f in fear_feats] + [2 for f in joy_feats] + [3 for f in sadness_feats]

    return tweets, emotions

def data_loader(tweets, emotions, num_samples, train_test_ratio = 0.75):
    # Data shuffle
    random.seed(42)
    zipped = list(zip(tweets, emotions))
    random.shuffle(zipped)
    tweets, emotions = zip(*(zipped[:num_samples]))
    tweets, emotions = np.asarray(tweets), np.asarray(emotions)
    # Train/test split
    num_data, num_train = len(emotions), int(len(emotions) * train_test_ratio)
    return (tweets[:num_train], emotions[:num_train]), (tweets[num_train:], emotions[num_train:])

def _create_bow(sentences, vectorizer=None, msg_prefix="\n", vocab = None):
    print("{} Bow construction".format(msg_prefix))
    if vectorizer is None:
        if vocab: 
            vectorizer = TfidfVectorizer(vocabulary = vocab)
            #vectorizer = CountVectorizer(vocabulary = vocab)
        else:
            vectorizer = TfidfVectorizer(stop_words = 'english', min_df = 3, max_df = 0.95, max_features = 2000)
            #vectorizer = CountVectorizer(stop_words = 'english', min_df = 3, max_df = 0.95, max_features = 2000)
        sentence_vectors = vectorizer.fit_transform(sentences)
    else:
        sentence_vectors = vectorizer.transform(sentences)
    return vectorizer, sentence_vectors.toarray(), sentences

class FeatureExtractor:
    def __init__(self, num_vocab, num_classes, scoreFactor):
        self.num_classes = num_classes
        self.num_vocab = num_vocab

        self.class_to_num_sentences = np.zeros(self.num_classes)
        self.class_and_word_to_counts = np.zeros((self.num_classes, self.num_vocab))
        self.score = np.zeros((self.num_classes, self.num_vocab))
        self.sf=scoreFactor
        self.log_prior = None
        self.log_likelihood = None

    def fit(self, bows, labels):
        # Compute self.class_to_num_sentences and self.class_and_word_to_counts
        for c in range(self.num_classes):
            self.class_to_num_sentences[c] = np.sum(labels == c)
        for bow, label in tqdm(zip(bows, labels), total=len(bows)):
            self.class_and_word_to_counts[label] += bow

        # Get log_prior and log_likelihood with these.
        self.log_prior = np.log(self.get_prior())
        self.log_likelihood = np.log(self.get_likelihood_with_smoothing())

    def get_prior(self):
        return self.class_to_num_sentences / np.sum(self.class_to_num_sentences)

    def get_likelihood_with_smoothing(self):
        likelihood_list = []
        for c in range(self.num_classes):
            likelihood = (self.class_and_word_to_counts[c] + 1) \
                         / (np.sum(self.class_and_word_to_counts[c]) + self.num_vocab)
            likelihood_list.append(likelihood)
        # likelihood_list = sorted(likelihood_list, key = lambda x: x[x][])
        nll = np.asarray(likelihood_list)
        return nll
    def feature_extract(self, max_features = None, feed_back = None):
        feature_list = []
        for c in range(self.num_classes):
            feature = []
            for w in range(len(self.log_likelihood[c])):
              score = max([self.log_likelihood[c][w] - self.log_likelihood[i][w] for i in range(self.num_classes)])
              feature.append((w, score))
            feature = sorted(feature, key = lambda x: x[1], reverse = True)[:max_features]
            feature_list.append(feature)
        for c in range(self.num_classes):
            for w, s in feature_list[c]:
                self.score[c][w] = s            
        if feed_back:
            ptr = 0
            for i, emotion in enumerate(feed_back):
                for word in emotion:
                    self.score[i][ptr] += self.sf
                    ptr += 1            
        return feature_list


def run(num_samples=10000, verbose=False, feed_back = [], scoreFactor=0):
    # Load the dataset
    tweets, emotions = get_data()
    (train_xs, train_ys), (val_xs, val_ys) = data_loader(tweets, emotions, num_samples, 0.75)
    if verbose:
        print("\n[Example of xs]: [\"{}...\", \"{}...\", ...]\n[Example of ys]: [{}, {}, ...]".format(
            train_xs[0][:70], train_xs[1][:70], train_ys[0], train_ys[1]))
        print("\n[Num Train]: {}\n[Num Test]: {}".format(len(train_ys), len(val_ys)))
    # Create bow representation of train set
    count_vectorizer, train_bows, _ = _create_bow(train_xs, msg_prefix="\n[Train]")
    myvocab = count_vectorizer.vocabulary_.copy()
    if not feed_back==[]:
        fb = feed_back[0] + feed_back[1] + feed_back[2] + feed_back[3]
        extend = len(fb)
        for word, id_ in myvocab.items():
            myvocab[word] = id_ + extend
        id_ = 0            
        for word in fb:
            myvocab[word] = id_
            id_ += 1
        count_vectorizer, train_bows, _ = _create_bow(train_xs, msg_prefix="\n[Train]", vocab = set(myvocab))
        myvocab = count_vectorizer.vocabulary_.copy()
    counted = len(count_vectorizer.get_feature_names())
    if verbose:
        print("\n[Vocab]: {} words".format(counted))
    fe = FeatureExtractor(num_vocab=counted, num_classes=4,scoreFactor=scoreFactor)
    fe.fit(train_bows, train_ys)
    if verbose:
        print("\n[FeatureExtractor] Training Complete")
    # id2word
    id2word = {}
    for w, id_ in myvocab.items():
      id2word[id_] = w
    # Extract features
    feature_list = fe.feature_extract(100, feed_back)
    print(fe.score.shape, len(myvocab))
    print(fe.score[:,10])
    print(id2word[10])
    title=["Word Cloud for Anger","Word Cloud for Fear","Word Cloud for Joy","Word Cloud for Sadness"]
    for i in range(4):
        top_feature = [(id2word[feat[0]], feat[1]) for feat in feature_list[i]]
        if verbose:
        	print(top_feature)
        	feat_dict = dict([(feat[0], np.exp(feat[1])) for feat in top_feature])
        	wordcloud = WordCloud(max_font_size=240,width=1920,height=1080,colormap="Set1",mode="RGBA",background_color=None)
        	wordcloud.generate_from_frequencies(feat_dict)
        	# Display the generated image:
        	plt.imshow(wordcloud, interpolation='bilinear')
        	plt.title(label=title[i])
        	plt.axis("off")
        	plt.show()        
    return fe, val_xs, val_ys, count_vectorizer

if __name__ == '__main__':
    run(feed_back=[["Hey"],[],[],[]])
