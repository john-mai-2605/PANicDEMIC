import csv
import random
import numpy as np
from tqdm import tqdm
import pandas as pd
import re
import extractor
import classifier
import nltk
    
def run(num_samples = 1000, num_sentences = 10):
        dates = ["2020-03-00 Coronavirus Tweets (pre 2020-03-12).csv","2020-03-12 Coronavirus Tweets.csv","2020-03-15 Coronavirus Tweets.csv","2020-03-20 Coronavirus Tweets.csv","2020-03-25 Coronavirus Tweets.csv","2020-03-28 Coronavirus Tweets.csv" ]
        data_list = []
        for date in dates:
                data_list.append(pd.read_csv(date, header=None, engine='python', skiprows=1, encoding = "utf-8"))
                data = map(lambda df: list(df[4][df[21] == 'en']), data_list)
        ##      negationArray = [negation.mark_negation(sent) for sent in val_xs]
        
        ext, val_xs, val_ys, count_vectorizer = extractor.run()
        clf = classifier.Classifier(ext.score, ext.log_prior, ext.num_classes)
        # Make validation bow
        val_bows_list = []
        for i in data:
                _, val_bow_data, sentences = extractor._create_bow(random.sample(i, num_samples), vectorizer=count_vectorizer, msg_prefix="\n[Validation]")
                val_bows_list.append((val_bow_data,sentences))
        # Evaluation
##      val_preds = clf.classify(val_xs, negationArray, count_vectorizer,ext.num_classes)
        val_data_list = []
        for x in val_bows_list:
                val_pred, val_scores = clf.classify(x[0])
                val_data_list.append((val_pred, val_scores,x[1]))

        # See trend of emotions over time
        result_data= []
        for i in val_data_list:
                fd = nltk.FreqDist(i[0])
                result_data.append(list(fd.items()))
        print(result_data)

        # See classification of random sentences from the first set
        
        check_list = list(zip(val_data_list[0][0],val_data_list[0][1], val_data_list[0][2]))
        check_anger = random.sample([item for item in check_list if item[0] == 0],num_sentences)
        print("ANGER:")
        for i in check_anger:
                print(i)
                
        check_fear = random.sample([item for item in check_list if item[0] == 1],num_sentences)
        print("FEAR:")
        for i in check_fear:
                print(i)
                
        check_joy = random.sample([item for item in check_list if item[0] == 2],num_sentences)
        print("JOY:")
        for i in check_joy:
                print(i)
                
        check_sadness = random.sample([item for item in check_list if item[0] == 3],num_sentences)
        print("SADNESS:")
        for i in check_sadness:
                print(i)

        

if __name__ == '__main__':
    run()
