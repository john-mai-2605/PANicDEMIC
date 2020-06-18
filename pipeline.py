import extractor
import classifier
import Cause_organizer as cause 
import check_classifier as check 

def run(progress = True, verbose = False):
    if progress:
        classifier.run(Covid = True, verbose = verbose)
        dates = ["../2020-04-19 Coronavirus Tweets.csv","../2020-04-21 Coronavirus Tweets.csv","../2020-04-22 Coronavirus Tweets.csv"]#,"../2020-04-24 Coronavirus Tweets.csv" ]
        # April overall Sun/Wed
        # dates = (["../2020-03-29 Coronavirus Tweets.csv","../2020-04-01 Coronavirus Tweets.csv","../2020-04-05 Coronavirus Tweets.csv","../2020-04-08 Coronavirus Tweets.csv"]
        #        +["../2020-04-{} Coronavirus Tweets.csv".format(i) for i in range(12,31,7)]
        #        +["../2020-04-{} Coronavirus Tweets.csv".format(i) for i in range(15,31,7)])


        # April overall Mon/Thu
        # dates = (["../2020-03-30 Coronavirus Tweets.csv","../2020-04-02 Coronavirus Tweets.csv","../2020-04-06 Coronavirus Tweets.csv","../2020-04-09 Coronavirus Tweets.csv"]
        #          +["../2020-04-{} Coronavirus Tweets.csv".format(i) for i in range(13,31,7)]
        #          +["../2020-04-{} Coronavirus Tweets.csv".format(i) for i in range(16,31,7)])            
        xA, xF, xJ, xS, cmFJS, cmAJS, cmAFS, cmAFJ = cause.run(verbose = verbose, dates = dates)
        classifier.run(Covid = True, verbose = verbose, feed_back = [cmFJS, cmAJS, cmAFS, cmAFJ])
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
    run()

