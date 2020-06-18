import csv
import random
import numpy as np
from tqdm import tqdm
import pandas as pd
import re
import extractor
import classifier
import nltk
from nltk.corpus import stopwords as stp
from gensim.models import Word2Vec, KeyedVectors
from nltk.cluster import KMeansClusterer
from sklearn.manifold import TSNE
from sklearn import cluster
import matplotlib.pyplot as plt 
def run(num_samples = 30000, num_sentences = 10, verbose = False, dates = ["../2020-04-19 Coronavirus Tweets.csv","../2020-04-21 Coronavirus Tweets.csv","../2020-04-22 Coronavirus Tweets.csv"]):
        # the list "avoid" contains manual filtering data
        avoid = ['...',"n't",'https']
        data_list = []
        for date in dates:
                data_list.append(pd.read_csv(date, header=None, engine='python', skiprows=1, encoding = "utf-8"))
        data = map(lambda df: list(df[4][df[21] == 'en']), data_list)
        # map(fun, iterable) returns function returns a map object(an iterator) of the results
        # after applying the given function to each item of a given iterable

        # val here means validation set. val_x means the validation tweet
        # and val_y means the validated emotion for the val_x(tweet)
        ext, val_xs, val_ys, count_vectorizer = extractor.run(verbose=verbose)
        clf = classifier.Classifier(ext.score, ext.log_prior, ext.num_classes)
        # Make validation bow(bag of words)
        val_bows_list = []
        random.seed(10)
        for i in data:
                _, val_bow_data, sentences = extractor._create_bow(random.sample(i, num_samples), vectorizer=count_vectorizer, msg_prefix="\n[Validation]")
                val_bows_list.append((val_bow_data,sentences))
        # Evaluation

        val_data_list = []
        for x in val_bows_list:
                val_pred, val_scores = clf.classify(x[0])
                val_data_list.append((val_pred, val_scores, x[1]))


        # See trend of emotions over time
        result_data= []
        for i in val_data_list:
                fd = nltk.FreqDist(i[0])
                result_data.append(list(fd.items()))

        print(result_data)

        # See classification of random sentences from the first set
        combined_preds,combined_scores,combined_tweets=[],[],[]
        for val_data in val_data_list:
                combined_preds=combined_preds+list(val_data[0])
                combined_scores=combined_scores+list(val_data[1])
                combined_tweets=combined_tweets+list(val_data[2])
        
        check_list = list(zip(combined_preds,combined_scores,combined_tweets))
        check_anger = sorted([item for item in check_list if item[0] == 0],key=lambda x:x[1],reverse=True)
        print("ANGER:", len(check_anger))
        for i in check_anger[:len(check_anger)//500]:
            print(i)

        #print(fearCFD.most_common(150))
        check_joy = sorted([item for item in check_list if item[0] == 2],key=lambda x:x[1],reverse=True)
        print("JOY:", len(check_joy))
        for i in check_joy[:len(check_joy)//500]:
            print(i)
                
        check_sadness = sorted([item for item in check_list if item[0] == 3],key=lambda x:x[1],reverse=True)
        print("SADNESS:", len(check_sadness))
        for i in check_sadness[:len(check_sadness)//500]:
            print(i)
        check_fear = sorted([item for item in check_list if item[0] == 1], key=lambda x:x[1],reverse=True)
        print("FEAR:", len(check_fear))
        for i in check_fear[:len(check_fear)//500]:
            print(i)
        model = Word2Vec([nltk.word_tokenize(twt[2].lower()) for twt in check_fear+check_sadness+check_joy+check_anger], size=50, workers=4, iter = 10)

        checks = [check_fear, check_sadness, check_joy, check_anger]

        title = ["Fear", "Sadness", "Joy", "Anger"]
        titlei =0
        result = []
        for check in checks:
            Words =[]
            for twt in check:
                Words += list(nltk.word_tokenize(twt[2].lower()))
            CFD = nltk.FreqDist(Words)

            # model = model.wv.save_word2vec_format()
            words = [w for w, fr in CFD.most_common(200) if (w not in stp.words('english') and len(w)>2 and w not in avoid)]
            result.append(words)
            vecs = [model[w] for w in words]
            tsne = TSNE(n_components=2)
            vecs_tsne = tsne.fit_transform(vecs)
            df = pd.DataFrame(vecs_tsne, index=words, columns=['x', 'y'])
            fig = plt.figure()
            fig.suptitle("Word Cluster for {}".format(title[titlei]),fontsize=25)
            ax = fig.add_subplot(1, 1, 1)
            ax.title.set_text(title[titlei])
            kcl = KMeansClusterer(5, nltk.cluster.util.cosine_distance, repeats = 50)
            Labels = kcl.cluster(vecs, assign_clusters=True)
            colors = []
            for i in Labels:
                if (i==0):
                        colors.append("r")
                elif (i==1):
                        colors.append("g")
                elif (i==2):
                        colors.append("y")
                elif (i==3):
                        colors.append("c")
                elif (i==4):
                        colors.append("m")
                else:
                        colors.append("k")
                            
            ax.scatter(df['x'], df['y'],marker=6, s=270, c=colors)
            titlei=titlei+1
            for word, pos in df.iterrows():
                ax.annotate(word, pos, fontsize=12) 
            if verbose:
                for j in range(10):
                    for i in range(len(vecs)):
                        if Labels[i] == j:
                            print(words[i])
                    print("\n")
                plt.show()      
        return result
if __name__ == '__main__':
    run(verbose = False)

