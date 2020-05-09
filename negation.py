import nltk
import re
nltk.download('punkt')
def mark_negation(sentence, double_neg=False):
  negation = {"n\'t",'no', 'not', 'none', 'never', 'nothing', 'nobody', 'nowhere',
              'neither', 'nor', "never","nowhere"}
  punct = re.compile(r"^[.:;!?]$")
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
        result.append(-1)
    elif neg_scope and punct.search(word):
      neg_scope = not neg_scope
      result.append(1)
    elif neg_scope and not punct.search(word):
      result.append(-1)
    else:
      result.append(1)
  return result

  