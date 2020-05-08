import nltk
from nltk.corpus import wordnet as wn
targetText = []

undesiredAdver=['by', 'across', 'around', 'but', 'between', 'on', 'over', 'after','forth','behind','together','under','about','so','through','inward','forward']
wierdlyNoun=['I', 'a', 'an', 'as', 'at', 'by', 'he', 'his', 'me', 'or', 'thou', 'us', 'who']

def isAdv(pos):
    if (pos=="RB" or pos=="RBT" or pos=="RBR"):
        return True
    return False
def isAdj(pos):
        if (pos=="JJ" or pos=="JJR" or pos=="JJT"):
        return True
    return False

def isIntensityMod(word): # P(IntensityMod | word) > threshold ??
    intModList=[very, so, slightly]
    if word in intModList:
        return True
    return True

def score(word):
    scorePresets={(very,0.9),(slightly,0.2),(overwhelmingly,1)}
    return 1
"""
Is this how Naive Bayes works?
Event1(very):Score=S1 | E2(slightly):S2 | E(so):S3 | E(overwhelmingly):S4
    
P(KnownEevent1|unknown)=P1            #P("very"|"mostly")
P(KnownEevent2|unknown)=P2
...

score = (P1*S1+P2*S2+...)/(P1+P2+...)
??
"""

emotionWords=[]
tags=nltk.pos_tag(targetText)
textLen=len(targetText)


# Intensity modifier for emotional words
for w in emotionWords:
    # Based on where the emotional word is
    location=targetText.index(w)

    # Is the word before the emotional word an intensity modifier?
    if location>0:
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
