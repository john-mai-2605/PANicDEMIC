import os
import tweepy
from tweepy import API 
from tweepy import Cursor
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import csv
 
import numpy as np
import pandas as pd

from dotenv import load_dotenv


# # # # TWITTER AUTHENTICATER # # # #
class TwitterAuthenticator():
    def authenticate_twitter_app(self):
        load_dotenv()
        
        consumer_key= os.environ.get("CONSUMER_KEY")
        consumer_secret= os.getenv("CONSUMER_SECRET")
        access_token= os.getenv("ACCESS_TOKEN")
        access_token_secret= os.getenv("ACCESS_TOKEN_SECRET")

        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        return auth
    
class TwitterClient():
    def __init__(self, twitter_user=None):
        self.auth = TwitterAuthenticator().authenticate_twitter_app()
        self.twitter_client = API(self.auth)

        self.twitter_user = twitter_user

    def get_user_timeline_tweets(self, num_tweets):
        tweets = []
        for tweet in Cursor(self.twitter_client.user_timeline, id=self.twitter_user).items(num_tweets):
            tweets.append(tweet)
        return tweets

    def get_friend_list(self, num_friends):
        friend_list = []
        for friend in Cursor(self.twitter_client.friends, id=self.twitter_user).items(num_friends):
            friend_list.append(friend)
        return friend_list

    def get_home_timeline_tweets(self, num_tweets):
        home_timeline_tweets = []
        for tweet in Cursor(self.twitter_client.home_timeline, id=self.twitter_user).items(num_tweets):
            home_timeline_tweets.append(tweet)
        return home_timeline_tweets

    def get_twitter_client_api(self):
        return self.twitter_client


# # # # TWITTER STREAMER # # # #
class TwitterStreamer():
    """
    Class for streaming and processing live tweets.
    """
    def __init__(self):
        self.twitter_autenticator = TwitterAuthenticator()    

    def stream_tweets(self, fetched_tweets_filename, hash_tag_list):
        # This handles Twitter authetification and the connection to Twitter Streaming API
        listener = TwitterListener(fetched_tweets_filename)
        auth = self.twitter_autenticator.authenticate_twitter_app() 
        stream = Stream(auth, listener)

        # This line filter Twitter Streams to capture data by the keywords: 
        stream.filter(track=hash_tag_list)


# # # # TWITTER STREAM LISTENER # # # #
class TwitterListener(StreamListener):
    """
    This is a basic listener that just prints received tweets to stdout.
    """
    def __init__(self, fetched_tweets_filename):
        self.fetched_tweets_filename = fetched_tweets_filename

    def on_data(self, data):
        try:
            
            with open(fetched_tweets_filename, 'wb') as fd:
                tweet_text = data.split(',"text":"')[1].split('","source"')[0]
                print(tweet_text)
                fd.write(str(tweet_text))
                fd.write("\n")
                fd.close()
                
            return True
        except BaseException as e:
            print("Error on_data %s" % str(e))
        return True
          
    def on_error(self, status):
        if status == 420:
            # Returning False on_data method in case rate limit occurs.
            return False
        print(status)


class TweetAnalyzer():
    """
    Functionality for analyzing and categorizing content from tweets.
    """
    def tweets_to_data_frame(self, tweets):
        df = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])

        df['id'] = np.array([tweet.id for tweet in tweets])
        df['lang'] = np.array([tweet.lang for tweet in tweets])
        df['date'] = np.array([tweet.created_at for tweet in tweets])
        df['geo'] = np.array([tweet.geo for tweet in tweets])
        df['likes'] = np.array([tweet.favorite_count for tweet in tweets])
        df['retweets'] = np.array([tweet.retweet_count for tweet in tweets])

        return df

def remove_unhashable_duplicates(items, key=None): 
    unique_items = set()
    for item in items:
        val = item if key is None else key(item) 
        if val not in unique_items:
            yield item 
            unique_items.add(val)
    
def GetTweets():

    auth = TwitterAuthenticator().authenticate_twitter_app()
    api = tweepy.API(auth)
    
    # JOY
    csvFile = open('result_joy.csv', 'a')

    csvWriter = csv.writer(csvFile)

    tweets = tweepy.Cursor(api.search,
                               q = "happy AND #covid-19",
                               tweet_mode='extended',
                               lang = "en").items(25)
    tweets_text = []
    for tweet in tweets:
        try:
            if tweet.retweeted_status.full_text not in tweets_text:
                csvWriter.writerow([tweet.created_at, tweet.retweeted_status.full_text])
                tweets_text.append(tweet.retweeted_status.full_text)
                print (tweet.retweeted_status.full_text)
        except AttributeError:
            if tweet.full_text not in tweets_text:
                csvWriter.writerow([tweet.created_at, tweet.full_text])
                tweets_text.append(tweet.full_text)
                print (tweet.full_text)
    csvFile.close()

        # FEAR
    csvFile = open('result_fear.csv', 'a')

    csvWriter = csv.writer(csvFile)

    tweets = tweepy.Cursor(api.search,
                               q = "afraid AND #covid-19",
                               tweet_mode='extended',
                               lang = "en").items(100)
    tweets_text = []
    for tweet in tweets:
        try:
            if tweet.retweeted_status.full_text not in tweets_text:
                csvWriter.writerow([tweet.created_at, tweet.retweeted_status.full_text])
                tweets_text.append(tweet.retweeted_status.full_text)
                print (tweet.retweeted_status.full_text)
        except AttributeError:
            if tweet.full_text not in tweets_text:
                csvWriter.writerow([tweet.created_at, tweet.full_text])
                tweets_text.append(tweet.full_text)
                print (tweet.full_text)
    csvFile.close()

        # SADNESS
    csvFile = open('result_sadness.csv', 'a')

    csvWriter = csv.writer(csvFile)

    tweets = tweepy.Cursor(api.search,
                               q = "sad AND #covid-19",
                               tweet_mode='extended',
                               lang = "en").items(100)
    tweets_text = []
    for tweet in tweets:
        try:
            if tweet.retweeted_status.full_text not in tweets_text:
                csvWriter.writerow([tweet.created_at, tweet.retweeted_status.full_text])
                tweets_text.append(tweet.retweeted_status.full_text)
                print (tweet.retweeted_status.full_text)
        except AttributeError:
            if tweet.full_text not in tweets_text:
                csvWriter.writerow([tweet.created_at, tweet.full_text])
                tweets_text.append(tweet.full_text)
                print (tweet.full_text)
    csvFile.close()

        # ANGER
    csvFile = open('result_anger.csv', 'a')

    csvWriter = csv.writer(csvFile)

    tweets = tweepy.Cursor(api.search,
                               q = "hate AND #covid-19",
                               tweet_mode='extended',
                               lang = "en").items(100)
    tweets_text = []
    for tweet in tweets:
        try:
            if tweet.retweeted_status.full_text not in tweets_text:
                csvWriter.writerow([tweet.created_at, tweet.retweeted_status.full_text])
                tweets_text.append(tweet.retweeted_status.full_text)
                print (tweet.retweeted_status.full_text)
        except AttributeError:
            if tweet.full_text not in tweets_text:
                csvWriter.writerow([tweet.created_at, tweet.full_text])
                tweets_text.append(tweet.full_text)
                print (tweet.full_text)
    csvFile.close()

    #COVID-19
    
    csvFile = open('result.csv', 'a')

    csvWriter = csv.writer(csvFile)

    tweets = tweepy.Cursor(api.search,
                               q = "covid-19",
                               tweet_mode='extended',
                               lang = "en").items(300)
    tweets_text = []
    for tweet in tweets:
        try:
            if tweet.retweeted_status.full_text not in tweets_text:
                csvWriter.writerow([tweet.created_at, tweet.retweeted_status.full_text])
                tweets_text.append(tweet.retweeted_status.full_text)
                print (tweet.retweeted_status.full_text)
        except AttributeError:
            if tweet.full_text not in tweets_text:
                csvWriter.writerow([tweet.created_at, tweet.full_text])
                tweets_text.append(tweet.full_text)
                print (tweet.full_text)
    csvFile.close()

 
if __name__ == '__main__':

    GetTweets()
##    twitter_client = TwitterClient()
##    tweet_analyzer = TweetAnalyzer()

##    api = twitter_client.get_twitter_client_api()
##    tweets = api.user_timeline(screen_name="realDonaldTrump", count=20)
##    df = tweet_analyzer.tweets_to_data_frame(tweets)

##    hash_tag_list = ["covid-19","virus", "happy", "joy"]
##    fetched_tweets_filename = "tweets.txt"
##
##    twitter_streamer = TwitterStreamer()
##    twitter_streamer.stream_tweets(fetched_tweets_filename, hash_tag_list)


