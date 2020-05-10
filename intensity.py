import nltk
from nltk.corpus import wordnet as wn


#undesiredAdver=['by', 'across', 'around', 'but', 'between', 'on', 'over', 'after','forth','behind','together','under','about','so','through','inward','forward']
#wierdlyNoun=['I', 'a', 'an', 'as', 'at', 'by', 'he', 'his', 'me', 'or', 'thou', 'us', 'who']

def isAdv(pos):
    if (pos=="RB" or pos=="RBT" or pos=="RBR"):
        return True
    return False
def isAdj(pos):
    if (pos=="JJ" or pos=="JJR" or pos=="JJT"):
        return True
    return False

def isIntensify(word): 
    intenseList=["very", "so","really","fucking","too","'too","well","severe","extremely","desperatily","dearly"]
    if word in intenseList:
        return True
    return False

def isDiminish(word):
    diminishList=["slightly", "bit","little"]
    if word in diminishList:
        return True
    return False

def score(word):
    scorePresets={(very,0.9),(slightly,0.2),(overwhelmingly,1)}
    return 1

def intensityScores(tweet):
    targetText=nltk.Text([w.lower() for w in nltk.word_tokenize(tweet)])
    tags=nltk.pos_tag(targetText)
    textLen=len(tags)
    result=[1 for i in range(len(tags))]
    
    for i in range(textLen-1):
        w=tags[i][0]
        wTag=tags[i][1]
        #if((isAdv(wTag) or isAdj(wTag)) and isIntensify(w)):
        if(isIntensify(w)):
            result[i+1]=result[i]*2
        #elif ((isAdv(wTag) or isAdj(wTag)) and isDiminish(w)):
        elif (isDiminish(w)):
            result[i+1]=result[i]/2
    return result
       
"""
emotionWords=[]
tags=nltk.pos_tag(targetText)
textLen=len(targetText)


# Intensity modifier for emotional words
for w in emotionWords:
    # Based on where the emotional word is
    location=targetText.index(w)

    # Is the word before the emotional word an intensity modifier?
    if location>0: # This is just
        investigate=targetText[location-1]
        invTag=tags[location-1][1]
        if (isAdv(invTag) or isAdj(invTag)) and isIntensityMod(investigate):
            if location>1:
                if targetText[location-2].lower()=="not":
                    -score(investigate)
                else:
                    score(investigate)

    # Is the word after the emotional word an intensity modifier?
    if location<textLen-1:
        investigate=targetText[location+1]
        invTag=tags[location+1][1]
        if (isAdv(invTag) or isAdj(invTag)) and isIntensityMod(investigate):
            score(investigate)
"""
