import extractor
import classifier
import Cause_organizer as cause 
import check_classifier as check
from pickle import dump,load

def run(progress = True, verbose = False, loadFile=False,printtweets=False,causeFilename="causeSunWedFri",outputDivider=900,produceResult=False,chunkScatter=False):
    if progress:
        classifier.run(Covid = True, verbose = verbose) #this gets the reference accuracy


        """
        various file input options as [dates]
        """
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


        # This part handles the loading/saving of the cause file for feedback usage.
        # Because the cause.run(heavily running code) doesn't run when you loadFile=True, this would be helpful.
        # If you don't have the files or don't have computation power, set loadFile=True and use the preset cause.pkl.
        if loadFile:
            loading=open(causeFilename+".pkl",'rb')
            xA, xF, xJ, xS, cmFJS, cmAJS, cmAFS, cmAFJ,_A,_F,_J,_S,cm4=load(loading)
            loading.close()
        else:
            xA, xF, xJ, xS, cmFJS, cmAJS, cmAFS, cmAFJ,_A,_F,_J,_S,cm4 = cause.run(verbose = verbose, dates = dates,printtweets=printtweets,chunkScatter=chunkScatter)
            saving=open(causeFilename+".pkl","wb")
            dump((xA, xF, xJ, xS, cmFJS, cmAJS, cmAFS, cmAFJ,_A,_F,_J,_S,cm4),saving,-1)
            saving.close()

        
        # This part shows you the accuracy information. 
        """ Feedback sandbox examples
        set scorefactor => sf
        sf = 0.2 : exclusive cause reinforce
        classifier.run(Covid = True, verbose = verbose, feed_back = [xA, xF, xJ, xS],sf=0.2)
        sf = -0.4 : non-cause deduction *(Sun/Wed => 0.84)
        classifier.run(Covid = True, verbose = verbose, feed_back = [cmFJS, cmAJS, cmAFS, cmAFJ],sf=-0.4)
        sf = -0.2 : inclusive cause reinforce *(Sun/Wed/Fri => 0.85)
        classifier.run(Covid = True, verbose = verbose, feed_back = [_A,_F,_J,_S],sf=-0.2)
        sf = ? : inclusive cause reinforce
        classifier.run(Covid = True, verbose = verbose, feed_back = [cm4,cm4,cm4,cm4],sf=0)"""
        fb=[cmFJS, cmAJS, cmAFS, cmAFJ]
        scoreFactor=-0.4
        classifier.run(Covid = True, verbose = verbose, feed_back = fb,sf=scoreFactor)
        # ^this part only tries the feedback on evaluation. This is just to show how accurate the classifier we will use on the bottom will be.
        # vThe real work is right below.


        #This part produces result(ex: 03-00 - Anger:4000, Fear: 1000, ...).
        if loadFile and produceResult:
            dateChunks=[# weekly analysis
                #["../2020-03-00 Coronavirus Tweets (pre 2020-03-12).csv"],
                #["../2020-03-12 Coronavirus Tweets.csv"],
                #["../2020-03-15 Coronavirus Tweets.csv"],
                #["../2020-03-00 Coronavirus Tweets (pre 2020-03-12).csv"]+["../2020-03-12 Coronavirus Tweets.csv"]+["../2020-03-15 Coronavirus Tweets.csv"],
                #["../2020-03-20 Coronavirus Tweets.csv"],
                #["../2020-03-25 Coronavirus Tweets.csv"],
                #["../2020-03-28 Coronavirus Tweets.csv"],
                #["../2020-03-29 Coronavirus Tweets.csv"],
                #["../2020-03-25 Coronavirus Tweets.csv","../2020-03-28 Coronavirus Tweets.csv","../2020-03-29 Coronavirus Tweets.csv"],
                #["../2020-03-30 Coronavirus Tweets.csv","../2020-03-31 Coronavirus Tweets.csv"]
                #+["../2020-04-0{} Coronavirus Tweets.csv".format(i) for i in range(1,6)],
                ["../2020-04-0{} Coronavirus Tweets.csv".format(i) for i in range(6,10)]
                +["../2020-04-{} Coronavirus Tweets.csv".format(i) for i in range(10,13)],
                ["../2020-04-{} Coronavirus Tweets.csv".format(i) for i in range(13,20)],
                ["../2020-04-{} Coronavirus Tweets.csv".format(i) for i in range(20,27)],
                ["../2020-04-{} Coronavirus Tweets.csv".format(i) for i in range(27,31)]
                    ]

            # This part will run and get the percentage informations.
            # REMEMBER, you should have all the files listed in dateChunks to run this part.
            # If you don't have them, set produceResult = False.
            for d in dateChunks:
                check.run(dates=d,verbose=False,
                          outDeminish=outputDivider,
                          feedback=fb,num_samples=3375,printtweets=printtweets)
            # Sidenote! The return of check is the cause chunks in list form from the dataset.
            #       -go to a for loop in check_classifier on line 106 ("for check in checks") for more info.
            #       -the check in checks are tweets as list in each emotions.(The exact format can be learned from line 81 to 96 check_classifier.py)
        
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
    run(verbose=False,loadFile=True,printtweets=True,produceResult=True, chunkScatter=False, causeFilename="((cm,-0.4)=0.84)causeSunWed",outputDivider=900)
    
"""
The [loadFile] field determines whether you are saving the causes or loading the causes.

The [printtweets] field determines whether you want the example tweets when doing cause saving.
    
The [produceResult] field determines whether the tweets counts(ex: Anger : 3000, Fear : 1000...) will be displayed.
    -This only takes effect when [loadFile] is True.
    -It should be put to False if you only want the accuracy result.
    
The [chunkScatter] field determines whether to plot the word scatters or not.
    -This only takes effect when [loadFile] is False.
    
The [causeFilename] field is the name of the file you are going to save/load.
    -BE CAREFUL not to overload the existing file.(Don't worry if you do, it's recoverable.)
    
The [outputDivider] is used to deminish the number of example tweets(when printtweets = True).
    -If set to 900, 9000 tweets result will only show top 10 tweets
    
To test out different cause feedback settings, adjust fb(for feedback causes) and scoreFactor on line 65 and 66.
"""
