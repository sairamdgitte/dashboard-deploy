import mysql.connector
import twint
import nltk, numpy as np
import string
from string import punctuation
import re
from mysql.connector import Error
from sqlalchemy import create_engine
import pandas as pd
# import modin.pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from deep_translator import GoogleTranslator
from sqlalchemy.dialects.mysql import insert
import time
from datetime import datetime
from dateutil import parser

import nest_asyncio
nest_asyncio.apply()

# Load in the cities
cities = pd.read_csv('au.csv')

# Translator object
translated = GoogleTranslator(source='auto', target='en')

# Sentiment analyzer object
sid = SentimentIntensityAnalyzer()

# # Porter stemmer
# ps = nltk.PorterStemmer()

# # Lemmatizer
# wn = nltk.WordNetLemmatizer()

# Stopword removal
stopword = nltk.corpus.stopwords.words('english')

# Stopword removal
stopword = nltk.corpus.stopwords.words('english')
words = set(nltk.corpus.words.words())


def clean_text(text):
    # Removing emojis, symbols, maps, flags etc
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"

                               "]+", flags=re.UNICODE)

    text = emoji_pattern.sub(r'', text)

    # Lower case
    text = text.lower( )

    # Removing URLs with a regular expression
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    text = url_pattern.sub(r'', text)

    # Remove Emails
    text = re.sub('\S*@\S*\s?', '', text)

    # Remove words with numbers and numbers
    text = re.sub('\w*[0-9]\w*\s?', '', text)

    # Remove #'s
    text = re.sub("[$&+,:;=?@#|'<>.^*()%!_-]?", '', text)

    # Remove new line characters
    text = re.sub('\s+', ' ', text)

    text = re.sub('[^a-zA-z0-9\s]', '', text)
    text = re.sub('[0-9]', '', text)

    text = re.sub('rt', '', text)

    # Remove distracting single quotes
    text = re.sub("\'", "", text)
    text = re.split('\W+', text)  # tokenization

    # Stopword removal
    text = [word for word in text if word not in stopword]  # remove stopwords and stemming

    # Remove non-english words / sentances
    text = " ".join(w for w in text if w.lower( ) in words or not w.isalpha( ))

    # Strip the leading spaces
    text = text.lstrip( ).rstrip( )

    if len(text.split( )) > 0:
        #         text = translated.translate(text)
        return text
    else:
        return np.nan


def translation(parm):
    a = translated.translate_batch(parm)

    return a



def create_db_connection(host_name, user_name, user_password, db_name):
    connection = None
    try:
        connection = mysql.connector.connect(
        host=host_name,
        user=user_name,
        passwd=user_password,
        database=db_name
        )
        print("MySQL Database connection successul")
    except Error as err:
        print(f"Error: '{err}'")
    return connection


def execute_query(connection, query):
    cursor = connection.cursor(buffered=True)
    try:
        cursor.execute(query)
        connection.commit( )
        print("Query successful")
    except Error as err:
        print(f"Error: '{err}'")
        pass

pw = "Sentiment"
connection = create_db_connection("database.cyg6qupmqicu.ap-southeast-2.rds.amazonaws.com", "admin", pw, "HTL_SENTIMENT")

df = pd.read_sql("SELECT * FROM tweet;", con=connection)
connection.close()


# if df.shape[0] == 0:
#     drop_table = """
#     DROP TABLE tweet;
#     """
#     execute_query(connection, drop_table)
#
#     create_tweet_table = """
#     CREATE TABLE tweet (
#     id VARCHAR(100),
#     original_tweet MEDIUMTEXT NOT NULL,
#     tweet MEDIUMTEXT NOT NULL,
#     lat DECIMAL(8,5),
#     lng DECIMAL(8,5),
#     place VARCHAR(50),
#     date VARCHAR(50),
#     sentiment INT(50),
#     PRIMARY KEY (`id`,`date`)
#     );
#     """
#     execute_query(connection, create_tweet_table)





date_diff = datetime.today() - parser.parse(df.date.max().split(" ")[0])



# Keywords to be searched
keywords = ['medicare', 'health', 'omicron', 'medical', 'wellness', \
            'medicine', 'vaccine', 'hospital', 'doctor', 'emergency room', \
            'physician', 'cardiologist', 'mammogram', 'oncologist', 'cancer', 'clinic', \
               'Primary care']

# connection = create_db_connection("database.cyg6qupmqicu.ap-southeast-2.rds.amazonaws.com", "admin", pw, "HTL_SENTIMENT")

if date_diff.days > 0:
    df1 = pd.DataFrame()
    for i in keywords:

        for j in cities.city.values[0:20].tolist():
            print('{}:{}'.format(j, i))
            c = twint.Config()

            c.Search = i
        #     c.Seacrh = "medicare OR health OR omicron OR medical OR wellness OR medicine OR vaccine"
        #     c.Search = "health"
            c.Geo = '{},{},300km'.format(cities[cities['city']==j].iloc[0,1],cities[cities['city']==j].iloc[0,2])
            c.Lang = 'en'
            c.Store_object = True
            c.Count = True
            c.Hide_output = True
            # c.Since = df.date.max().split(" ")[0]
            c.Since = df.date.max()
            c.Until = datetime.today().strftime("%Y-%m-%d")
            # c.Until = '2022-12-31'
            c.Filter_retweets = True
            c.Pandas = True

            twint.run.Search(c)

            # Sleep time added
            time.sleep(2)

            twint.storage.panda.Tweets_df.place = cities[cities['city']==j].iloc[0,5]

        #     df = pd.concat([df, twint.storage.panda.Tweets_df])
            df_twint = twint.storage.panda.Tweets_df.copy()
            df_twint['lat'] = cities[cities['city']==j].iloc[0,1]
            df_twint['lng'] = cities[cities['city']==j].iloc[0,2]

            if df_twint.shape[0] > 0:
                df_twint = df_twint.drop_duplicates('tweet')

                df_twint = df_twint[["id", "tweet", "lat", "lng", "place", "date"]]

                df_twint['sentiment_tweet'] = df_twint.tweet.apply(lambda x: clean_text(x))
                df_twint = df_twint.dropna(subset=['sentiment_tweet'])
                df_twint['sentiment_tweet'] = translation(df_twint.sentiment_tweet.values.tolist())

                df_twint['sentiment'] = df_twint['sentiment_tweet'].apply(lambda x: 1 if sid.polarity_scores(x)['compound'] >= 0.05 else -1 if sid.polarity_scores(x)['compound'] <= -0.05 else 0)
                df_twint = df_twint.rename(columns={'tweet': 'original_tweet', \
                                        'sentiment_tweet':'tweet'})[["id","original_tweet","tweet", "lat", "lng", "place", "date", "sentiment"]]
                df1 = pd.concat([df1, df_twint])
                    
engine = create_engine('mysql+pymysql://admin:Sentiment@database.cyg6qupmqicu.ap-southeast-2.rds.amazonaws.com/HTL_SENTIMENT')


def insert_on_duplicate(table, conn, keys, data_iter):
    insert_stmt = insert(table.table).values(list(data_iter))
    on_duplicate_key_stmt = insert_stmt.on_duplicate_key_update(insert_stmt.inserted)
    conn.execute(on_duplicate_key_stmt)
df1.to_csv('new_data.csv')
df1.to_sql('tweet', con=engine, if_exists='append', index=False, chunksize = 1000, method=insert_on_duplicate)
