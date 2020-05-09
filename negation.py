import nltk
from copy import deepcopy
nltk.download('punkt')
# Scoring negation
def score_negation(sentence):
  # Quasi negative words
  negation = {"n\'t",'no', 'not', 'none', 'never', 'nothing', 'nobody', 'nowhere',
              'neither', 'nor', "never","nowhere"}
  quasi_neg = {'hardly','rarely', 'scarcely', 'seldom', 'barely', 'few', 'little'}
  neg_implication = {'fail', 'without', 'beyond', 'until', 'unless', 'lest', 
                    'ignorant', 'refuse', 'neglect', 'absence',
                    'instead of','despite','uh-uh'}
  tokens = [word.lower() for word in nltk.word_tokenize(sentence)]
  print(tokens)
  result_score= []
  for word in tokens:
    if (word in negation) or (word in neg_implication):
      result_score.append(-1)
    elif word in quasi_neg:
      result_score.append(0.5)
    else:
      result_score.append(1)
  return result_score