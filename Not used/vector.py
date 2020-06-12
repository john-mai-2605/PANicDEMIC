import math
import re
import pandas as pd

docA = "So my Indian Uber driver just called someone the N word. If I wasn't in a moving vehicle I'd have jumped out #disgusted"
docB = "Don't join @BTCare they put the phone down on you, talk over you and are rude. Taking money out of my acc willynilly! #fuming"

class tfidfVec():
    def __init__(self, text_list):
      self.text_list = text_list
      self.bag_list = []
      self.word_dict_list = []
      self.word_set =set()

    def TF(word_dict, bag):
        tf_dict = {}
        for word, count in word_dict.items():
            tf_dict[word] = count/float(len(bag))
        return tf_dict

    def IDF(doc_list): 
        idf_dict = dict.fromkeys(doc_list[0].keys(), 0)
        for doc in doc_list:
            for word, val in doc.items():
                if val > 0:
                    idf_dict[word] += 1   
        for word, val in idf_dict.items():
            idf_dict[word] = 1.0 + math.log10(len(doc_list) / float(val))
        return idf_dict

    def TFIDF(tf, idfs):
        tfidf = {}
        for word, val in tf.items():
          tfidf[word] = val*idfs[word]
        return tfidf 

    def vectorize(self): 
      for text in self.text_list:
        self.bag_list.append(re.split('\W', text))
      self.word_set = set().union(*self.bag_list)
      for bag in self.bag_list:
        self.word_dict_list.append(dict.fromkeys(self.word_set, 0))
        for word in bag:
          self.word_dict_list[-1][word] += 1
      
      tf_list = [TF(word_dict, self.bag_list[idx]) for idx, word_dict in enumerate(self.word_dict_list)]
      # print(tf_list)
      idfs = IDF(self.word_dict_list)
      tf_idf_list = [TFIDF(tf, idfs) for tf in tf_list]
      # print(tf_idf_list)
      return pd.DataFrame(tf_idf_list)
