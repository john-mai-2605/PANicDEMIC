import extractor
import classifier
import Cause_organizer as cause 
import check_classifier as check
from pickle import dump,load

def run(progress = True, verbose = False, loadFile=False,printtweets=False,causeFilename="causeSunWedFri",outputDivider=900,produceResult=False):
    if progress:
        classifier.run(Covid = True, verbose = verbose)
        #dates = ["../2020-04-19 Coronavirus Tweets.csv","../2020-04-21 Coronavirus Tweets.csv","../2020-04-22 Coronavirus Tweets.csv"]#,"../2020-04-24 Coronavirus Tweets.csv" ]
        # April overall Sun/Wed
        dates = (["../2020-03-29 Coronavirus Tweets.csv","../2020-04-01 Coronavirus Tweets.csv","../2020-04-05 Coronavirus Tweets.csv","../2020-04-08 Coronavirus Tweets.csv"]
                +["../2020-04-{} Coronavirus Tweets.csv".format(i) for i in range(12,31,7)]
                +["../2020-04-{} Coronavirus Tweets.csv".format(i) for i in range(15,31,7)])


        # April overall Mon/Thu
        #dates = (["../2020-03-30 Coronavirus Tweets.csv","../2020-04-02 Coronavirus Tweets.csv","../2020-04-06 Coronavirus Tweets.csv","../2020-04-09 Coronavirus Tweets.csv"]
        #          +["../2020-04-{} Coronavirus Tweets.csv".format(i) for i in range(13,31,7)]
        #          +["../2020-04-{} Coronavirus Tweets.csv".format(i) for i in range(16,31,7)])

        
        # April overall Sun/Wed/Fri
        #dates = (["../2020-03-29 Coronavirus Tweets.csv","../2020-04-01 Coronavirus Tweets.csv","../2020-04-03 Coronavirus Tweets.csv","../2020-04-06 Coronavirus Tweets.csv","../2020-04-08 Coronavirus Tweets.csv"]
        #          +["../2020-04-{} Coronavirus Tweets.csv".format(i) for i in range(10,31,7)]
        #          +["../2020-04-{} Coronavirus Tweets.csv".format(i) for i in range(12,31,7)]
        #          +["../2020-04-{} Coronavirus Tweets.csv".format(i) for i in range(15,31,7)])

        # April overall Mon/Thu/Sat
        #dates = (["../2020-03-30 Coronavirus Tweets.csv","../2020-04-02 Coronavirus Tweets.csv","../2020-04-04 Coronavirus Tweets.csv","../2020-04-06 Coronavirus Tweets.csv","../2020-04-09 Coronavirus Tweets.csv"]
        #          +["../2020-04-{} Coronavirus Tweets.csv".format(i) for i in range(13,31,7)]
        #          +["../2020-04-{} Coronavirus Tweets.csv".format(i) for i in range(16,31,7)]
        #          +["../2020-04-{} Coronavirus Tweets.csv".format(i) for i in range(11,31,7)])

        
        if loadFile:
            loading=open(causeFilename+".pkl",'rb')
            xA, xF, xJ, xS, cmFJS, cmAJS, cmAFS, cmAFJ,_A,_F,_J,_S,cm4=load(loading)
            loading.close()
        else:
            xA, xF, xJ, xS, cmFJS, cmAJS, cmAFS, cmAFJ,_A,_F,_J,_S,cm4 = cause.run(verbose = verbose, dates = dates)
            saving=open(causeFilename+".pkl","wb")
            dump((xA, xF, xJ, xS, cmFJS, cmAJS, cmAFS, cmAFJ,_A,_F,_J,_S,cm4),saving,-1)
            saving.close()

            
        """ Feedback sandbox examples
        set scorefactor => sf
        sf = 0.2 : exclusive cause reinforce
        classifier.run(Covid = True, verbose = verbose, feed_back = [xA, xF, xJ, xS],sf=0.2)
        sf = -0.4 : non-cause deduction *(Sun/Wed => 0.84)
        classifier.run(Covid = True, verbose = verbose, feed_back = [cmFJS, cmAJS, cmAFS, cmAFJ],sf=-0.4)
        sf = -0.2 : inclusive cause reinforce *(Sun/Wed/Fri => 0.85)
        classifier.run(Covid = True, verbose = verbose, feed_back = [_A,_F,_J,_S],sf=-0.2)
        sf = ? : inclusive cause reinforce
        classifier.run(Covid = True, verbose = verbose, feed_back = [cm4,cm4,cm4,cm4],sf=0)
        """
        
        classifier.run(Covid = True, verbose = verbose, feed_back = [cmFJS, cmAJS, cmAFS, cmAFJ],sf=-0.4)

        if loadFile and produceResult:
            dateChunks=[# weekly analysis
                ["../2020-03-00 Coronavirus Tweets (pre 2020-03-12).csv"],
                ["../2020-03-12 Coronavirus Tweets.csv"],
                ["../2020-03-15 Coronavirus Tweets.csv"],
                ["../2020-03-00 Coronavirus Tweets (pre 2020-03-12).csv"]+["../2020-03-12 Coronavirus Tweets.csv"]+["../2020-03-15 Coronavirus Tweets.csv"],
                ["../2020-03-20 Coronavirus Tweets.csv"],
                ["../2020-03-25 Coronavirus Tweets.csv"],
                ["../2020-03-28 Coronavirus Tweets.csv"],
                ["../2020-03-29 Coronavirus Tweets.csv"],
                ["../2020-03-25 Coronavirus Tweets.csv","../2020-03-28 Coronavirus Tweets.csv","../2020-03-29 Coronavirus Tweets.csv"],
                ["../2020-03-30 Coronavirus Tweets.csv","../2020-03-31 Coronavirus Tweets.csv"]
                +["../2020-04-0{} Coronavirus Tweets.csv".format(i) for i in range(1,6)],
                ["../2020-04-0{} Coronavirus Tweets.csv".format(i) for i in range(6,10)]
                +["../2020-04-{} Coronavirus Tweets.csv".format(i) for i in range(10,13)],
                ["../2020-04-{} Coronavirus Tweets.csv".format(i) for i in range(13,20)],
                ["../2020-04-{} Coronavirus Tweets.csv".format(i) for i in range(20,27)],
                ["../2020-04-{} Coronavirus Tweets.csv".format(i) for i in range(27,31)]
                    ]

            for d in dateChunks:
                check.run(dates=d,verbose=False,loadFile4result=True,printtweets=printtweets,outDeminish=outputDivider,loadFilename=causeFilename)
        
        
    else:
        # the list "dates" contain the path of the tweets' files
        
        # original
        # dates = ["../2020-04-19 Coronavirus Tweets.csv","../2020-04-21 Coronavirus Tweets.csv","../2020-04-22 Coronavirus Tweets.csv"]#,"../2020-04-24 Coronavirus Tweets.csv" ]

        # April 16~30
        #dates = ["../2020-04-{} Coronavirus Tweets.csv".format(i) for i in range(16,31)]

        # April 01~15
        #dates = (["../2020-04-{} Coronavirus Tweets.csv".format(i) for i in range(10,16)]+["../2020-04-0{} Coronavirus Tweets.csv".format(i) for i in range(1,10)])        
        check_classifier.run(verbose = verbose)
if __name__ == '__main__':
    run(loadFile=True,printtweets=False,produceResult=True, causeFilename="((cm,-0.4)=0.84)causeSunWed",outputDivider=900)
    
"""To try different accuracy settings
edit line 59(or if not sure, line with classiferi.run(...)
*important*edit line33 in check_classifier.py
the [feed_back] field takes what set of causes you are going to apply score changes
the [sf] field takes how much scoring change you will apply(may be of any sign)
"""
