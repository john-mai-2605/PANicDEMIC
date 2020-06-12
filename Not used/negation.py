import csv
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import pandas as pd
import re
import extractor
import nltk
import re
def mark_negation(sentence, double_neg=False):
  negation = {"n\'t",'no', 'not', 'none', 'never', 'nothing', 'nobody', 'nowhere',
              'neither', 'nor', "never","nowhere"}
  punct = re.compile(r"^[.:;,!?]$")
  conjuction = {'after','although',
'as','as if','as long as',
'as much as',
'as soon as',
'as though',
'because',
'before',
'by the time',
'even if',
'even though',
'if',
'in order that',
'in case',
'in the event that',
'lest',
'now that',
'once',
'only',
'only if',
'provided that',
'since',
'so',
'supposing',
'that',
'than',
'though',
'till',
'unless',
'until', 'when',
'whenever',
'where',
'whereas',
'wherever',
'whether or not',
'while'
}
  tokens = [word.lower() for word in nltk.word_tokenize(sentence)]
  result=[]
  neg_scope = False
  for i,word in enumerate(tokens):
    if word in negation:
      if not neg_scope or (neg_scope and double_neg):
        neg_scope = not neg_scope
        result.append(1)
        continue
      else:
        result.append(0.5)
    elif neg_scope and (punct.search(word) or (word in conjuction)):
      neg_scope = not neg_scope
      result.append(1)
    elif neg_scope and not (punct.search(word) or (word in conjuction)):
      result.append(0.5)
    else:
      result.append(1)
  return result