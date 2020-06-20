from nltk import word_tokenize as wt
import check_classifier as cc
"""apr01="./April 01~15"
apr16="./April 16~30"
aproverall1="./April overall(Sun_Wed)/"
aproverall2="./April overall(Mon_Thu)/"
aproverall1t=" March 29~ April30(2 per week).txt"
aproverall2t=" March 30~ April30(2 per week).txt"
angerf =    open(aproverall1+"anger"+aproverall1t)
fearf =     open(aproverall1+"fear"+aproverall1t)
joyf =      open(aproverall1+"joy"+aproverall1t)
sadnessf =  open(aproverall1+"sadness"+aproverall1t)
"""

def common2(a,b):# if i in a and b
    common=[]
    for i in a:
        if i in b:
            common.append(i)
    return(common)
def common3(a,b,c): # if i in common(a,b) and c
    temp=common2(a,b)
    return common2(temp,c)

def common4(a,b,c,d): # if i in common(a,b,c) and d
    temp=common3(a,b,c)
    return common2(temp,d)

def excommon3(a,b,c,com4): # if i in common(a,b,c) and not in allcommon
    temp=common3(a,b,c)
    xcommon3=[]
    for i in temp:
        if i not in com4:
            xcommon3.append(i)
    return xcommon3

def excommon2(a,b,com3s): # if i in common(a,b) and not in com3s
    temp=common2(a,b)
    xcommon2=[]
    for i in temp:
        if i not in com3s:
            xcommon2.append(i)
    return xcommon2
def run(verbose = False,
        dates = ["../2020-04-19 Coronavirus Tweets.csv","../2020-04-21 Coronavirus Tweets.csv","../2020-04-22 Coronavirus Tweets.csv"],
        printtweets=False, chunkScatter=False
        ):
    cause_list=cc.run(verbose=verbose, dates = dates,printtweets=printtweets,chunkScatter=chunkScatter)
    anger=cause_list[0]
    fear=cause_list[1]
    joy=cause_list[2]
    sadness=cause_list[3]
    combined=anger+fear+joy+sadness
    com4=common4(anger,fear,joy,sadness)
    cmAFJ=excommon3(anger,fear,joy,com4)
    cmAFS=excommon3(anger,fear,sadness,com4)
    cmAJS=excommon3(anger,joy,sadness,com4)
    cmFJS=excommon3(fear,joy,sadness,com4)
    com3s=list(set(cmAFJ+cmAFS+cmAJS+cmFJS+com4))
    cmAF=excommon2(anger,fear,com3s)
    cmAJ=excommon2(anger,joy,com3s)
    cmAS=excommon2(anger,sadness,com3s)
    cmFJ=excommon2(fear,joy,com3s)
    cmFS=excommon2(fear,sadness,com3s)
    cmJS=excommon2(joy,sadness,com3s)
    cm=list(set(cmAF+cmAJ+cmAS+cmFJ+cmFS+cmJS))
    xA=excommon2(anger,anger,com3s+cm)
    xF=excommon2(fear,fear,com3s+cm)
    xJ=excommon2(joy,joy,com3s+cm)
    xS=excommon2(sadness,sadness,com3s+cm)
    _A=excommon2(anger,anger,com4)
    _F=excommon2(fear,fear,com4)
    _J=excommon2(joy,joy,com4)
    _S=excommon2(sadness,sadness,com4)
    if verbose:#under this part is just printing part(massive, I know)
        print("Common for all")
        for i in com4:
            print(i,end=", ")
        print()
        print()
        print("Exclusively common to A,F,J")
        for i in cmAFJ:
            print(i,end=", ")
        print()
        print()
        print("Exclusively common to A,F,S")
        for i in cmAFS:
            print(i,end=", ")
        print()
        print()
        print("Exclusively common to A,J,S")
        for i in cmAJS:
            print(i,end=", ")
        print()
        print()
        print("Exclusively common to F,J,S")
        for i in cmFJS:
            print(i,end=", ")
        print()
        print()
        print("Exclusively common to A,F")
        for i in cmAF:
            print(i,end=", ")
        print()
        print()
        print("Exclusively common to A,J")
        for i in cmAJ:
            print(i,end=", ")
        print()
        print()
        print("Exclusively common to A,S")
        for i in cmAS:
            print(i,end=", ")
        print()
        print()
        print("Exclusively common to F,J")
        for i in cmFJ:
            print(i,end=", ")
        print()
        print()
        print("Exclusively common to F,S")
        for i in cmFS:
            print(i,end=", ")
        print()
        print()
        print("Exclusively common to J,S")
        for i in cmJS:
            print(i,end=", ")
        print()
        print()
        print("Exclusively Anger")
        for i in xA:
            print(i,end=", ")
        print()
        print()
        print("Exclusively Fear")
        for i in xF:
            print(i,end=", ")
        print()
        print()
        print("Exclusively Joy")
        for i in xJ:
            print(i,end=", ")
        print()
        print()
        print("Exclusively Sadness")
        for i in xS:
            print(i,end=", ")
    return xA, xF, xJ, xS, cmFJS, cmAJS, cmAFS, cmAFJ,_A,_F,_J,_S,com4

if __name__ == '__main__':
    run(verbose = True)
