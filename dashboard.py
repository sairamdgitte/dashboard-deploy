import asyncio
import itertools
import json
import re
import time
from collections import Counter
from urllib.request import urlopen


import dash_bootstrap_components as dbc
import mysql.connector
import nltk
import numpy as np
import pandas as pd
import plotly.express as px  # (version 4.7.0 or higher)
import stylecloud
from dash import Dash, dcc, html  # pip install dash (version 2.0.0 or higher)
# from nltk.tokenize import word_tokenize
import dash_extensions as de
from datetime import datetime
import datetime as dt
import time
import itertools
import dash_daq as daq
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


# Lotties: Emil at https://github.com/thedirtyfew/dash-extensions

url = "./assets/thumbs-up.json"
url2 = "./assets/thumbs-down.json"
url3 = "./assets/twitter.json"
url4 = "./assets/keyboard.json"

options = dict(loop=True, autoplay=True, rendererSettings=dict(preserveAspectRatio='xMidYMid slice'))
options1 = dict(loop=False, autoplay=True, rendererSettings=dict(preserveAspectRatio='xMidYMid slice'))


def create_db_connection(host_name, user_name, user_password, db_name):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            database=db_name
        )
        print("MySQL Database connection successful")
    except Error as err:
        print(f"Error: '{err}'")
    return connection


pw = "Sentiment"

connection = create_db_connection("database.cyg6qupmqicu.ap-southeast-2.rds.amazonaws.com", "admin", pw,
                                  "HTL_SENTIMENT")


with urlopen('https://raw.githubusercontent.com/rowanhogan/australian-states/master/states.geojson') as response:
    counties = json.load(response)

state_id_map = {}
for i in counties['features']:
    state_id_map[i['properties']['STATE_NAME']] = i['id']



# server = app.server
df = pd.read_sql("SELECT * FROM tweet;", con=connection)
flat_list = list(itertools.chain.from_iterable(df.tweet.apply(lambda x: x.split(' '))))

# Create bigrams
bgs = nltk.bigrams(flat_list)

freq_count = nltk.FreqDist(bgs)

freq_count_df = pd.DataFrame(dict(freq_count).items( ))

freq_count_df = freq_count_df.sort_values(1, ascending=False).iloc[:20, :]

viz = df.groupby(by=['place', 'sentiment']).count( ).reset_index( )
pos_sentiment = viz[viz['sentiment'] == 1]
neg_sentiment = viz[viz['sentiment'] == -1]
neutral_sentiment = viz[viz['sentiment'] == 0]
pos_sentiment['place_id'] = pos_sentiment['place'].map(state_id_map)
neg_sentiment['place_id'] = neg_sentiment['place'].map(state_id_map)

# df['cleaned_tweet'] = df['tweet'].apply(lambda x: clean_text(x))
# pos_sentiment['log_tweet'] = np.log(pos_sentiment['tweet'])
# neg_sentiment['log_tweet'] = np.log(neg_sentiment['tweet'])

df = df[df.tweet != '']

# Create three Counter objects to store positive, negative and total counts
positive_counts = Counter( )
negative_counts = Counter( )
neutral_counts = Counter( )
total_counts = Counter( )

positive_tweets = df[df['sentiment'] == 1].tweet.values.tolist( )
negative_tweets = df[df['sentiment'] == -1].tweet.values.tolist( )
neutral_tweets = df[df['sentiment'] == 0].tweet.values.tolist( )

for i in range(len(positive_tweets)):
    for word in positive_tweets[i].split(" "):
        positive_counts[word] += 1
        total_counts[word] += 1

for i in range(len(negative_tweets)):
    for word in negative_tweets[i].split(" "):
        negative_counts[word] += 1
        total_counts[word] += 1

for i in range(len(neutral_tweets)):
    for word in neutral_tweets[i].split(" "):
        neutral_counts[word] += 1
        total_counts[word] += 1

pos_neg_ratios = Counter( )

# Calculate the ratios of positive and negative uses of the most common words
# Consider words to be "common" if they've been used at least 100 times
for term, cnt in list(total_counts.most_common( )):
    if (cnt > 100):
        pos_neg_ratio = positive_counts[term] / (float(negative_counts[term] + 1) + float(neutral_counts[term] + 1))
        pos_neg_ratios[term] = pos_neg_ratio


# Convert ratios to logs
for word, ratio in pos_neg_ratios.most_common( ):
    if ratio <= 0:
        pos_neg_ratios[word] = -np.log(1 / (ratio + 0.1))
    else:
        pos_neg_ratios[word] = np.log(ratio)

negative_intensity = pos_neg_ratios.most_common( )[-20:]
positive_intensity = pos_neg_ratios.most_common( )[:20]

del pos_neg_ratios


total_words = sum(list(dict(positive_counts).values())) + sum(list(dict(negative_counts).values())) + sum(list(dict(neutral_counts).values()))

del positive_counts
del negative_counts
del neutral_counts

# words most frequently seen in a review with a "NEGATIVE" label
neg_df = pd.DataFrame(list(negative_intensity))

neg_df['freq'] = neg_df[0].apply(lambda x: dict(total_counts)[x])

# del total_counts
neg_df = neg_df.iloc[:10, :].set_index(0)

neg_df = neg_df.reset_index( )

pos_df = pd.DataFrame(positive_intensity)
pos_df['freq'] = pos_df[0].apply(
    lambda x: dict(total_counts)[x] if x in dict(total_counts) else 0)


df['hashtags'] = df.original_tweet.apply(lambda x: list(set(re.findall('#[A-Za-z0-9_]+', x) )) if len(set(re.findall('#[A-Za-z0-9_]+', x) )) > 0 else [])
pos_tags = list(itertools.chain(*df[df['sentiment']==1]['hashtags'].to_list()))
neg_tags = list(itertools.chain(*df[df['sentiment']==-1]['hashtags'].to_list()))

stylecloud.gen_stylecloud(' '.join(pos_tags + neg_tags), colors=['#41B3A3', '#9d3f54'], size=(1024, 800),
                          background_color='#070914', icon_name='fas fa-hashtag', output_name='./assets/pos_cloud.png')

df['mentions'] = df.original_tweet.apply(lambda x: list(set(re.findall('@[A-Za-z0-9_]+', x) )) if len(set(re.findall('@[A-Za-z0-9_]+', x) )) > 0 else [])
# unique_mentions = len(set(list(itertools.chain(*df['mentions'].to_list()))))
pos_mentions = list(itertools.chain(*df[df['sentiment']==1]['mentions'].to_list()))
neg_mentions = list(itertools.chain(*df[df['sentiment']==-1]['mentions'].to_list()))
stylecloud.gen_stylecloud(' '.join(pos_mentions + neg_mentions), colors=['#41B3A3', '#9d3f54'], size=(1024, 800),
                          background_color='#070914', icon_name='fas fa-at', output_name='./assets/at_cloud.png')
# print(image_colors)
pos_df = pos_df.iloc[:10, :]
# print(pos_df)
# print(neg_df)

# def update_graph(df, color_name, tweet):
#     print(df)
#     fig_Heterogeneity = px.choropleth_mapbox(df,
#                                              locations="place_id",
#                                              geojson=counties,
#                                              color="tweet",
#                                              mapbox_style="carto-darkmatter",  # carto-darkmatter",
#                                              hover_name='place',
#                                              hover_data={'place_id': False},
#                                              opacity=0.8, color_continuous_midpoint=0,
#                                              color_continuous_scale=eval('px.colors.sequential.' + color_name),
#                                              labels={'EC': 'Environmental Heterogeneity'},
#                                              center={"lat": -25.2744, "lon": 133.7751},
#                                              range_color=(0, df.tweet.max( )),
#                                              zoom=2,
#                                              height=300)
#     fig_Heterogeneity.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, paper_bgcolor="#262626", plot_bgcolor='#262626',
#                                     font={
#                                         "color": "white"})
#     fig_Heterogeneity.update_geos(fitbounds="locations", visible=False)

#     return fig_Heterogeneity
def graph(df, color_name):
    df_state_sentiment = pd.DataFrame()
    for i in df.place.unique():
        df_state_sentiment = pd.concat([df_state_sentiment, df[df['place']==i]['sentiment'].value_counts().to_frame().rename(columns={'sentiment': i}).T]) 
    df_state_sentiment['total_sentiment'] = -df_state_sentiment[-1]+df_state_sentiment[1]
    df_state_sentiment['percentage'] = (df_state_sentiment['total_sentiment']/df_state_sentiment['total_sentiment'].sum()*100)
    df_state_sentiment.reset_index(inplace=True)
    print(df_state_sentiment)
    df_state_sentiment['place_id'] = df_state_sentiment['index'].map(state_id_map)

    fig_Heterogeneity = px.choropleth_mapbox(df_state_sentiment,
                                             locations="place_id",
                                             geojson=counties,
                                             color="percentage",
                                             mapbox_style="carto-darkmatter",  # carto-darkmatter",
                                             hover_name='index',
                                             hover_data={'place_id': False},
                                             opacity=0.8, color_continuous_midpoint=0,
                                             color_continuous_scale=eval('px.colors.sequential.' + color_name),
                                             labels={'EC': 'Environmental Heterogeneity'},
                                             center={"lat": -25.2744, "lon": 133.7751},
                                             range_color=(0, 100),
                                             zoom=2,
                                             height=300)
    fig_Heterogeneity.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, paper_bgcolor="#262626", plot_bgcolor='#262626',
                                    font={
                                        "color": "white"})
    fig_Heterogeneity.update_geos(fitbounds="locations", visible=False)

    return fig_Heterogeneity

def uni_gram_graph(df, color_name):
    bar_fig = px.bar(df[[0, 'freq']].sort_values(by='freq', ascending=True),
                     x='freq', y=0, orientation='h', text='freq', color='freq',
                     color_continuous_scale=color_name,
                     labels={
                         "0": "Unigrams",
                         "freq": "Frequency",
                         "freq": "Frequency"
                     },
                    height=150
                     )

    bar_fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    bar_fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), uniformtext_minsize=4, uniformtext_mode='hide',
                          plot_bgcolor='#070914', paper_bgcolor='#070914',
                          font={
                              "color": "white"
                          })
    return bar_fig


def data_for_bigram(df, sentiment='pos'):
    if sentiment == 'pos':
        df1 = df[df['sentiment'] == 1]
    else:
        df1 = df[df['sentiment'] == 0]

    flat_list = list(itertools.chain.from_iterable(df1.tweet.apply(lambda x: x.split(' '))))

    # Create bigrams
    bgs = nltk.bigrams(flat_list)

    freq_count = nltk.FreqDist(bgs)

    freq_count_df = pd.DataFrame(dict(freq_count).items( ))

    freq_count_df = freq_count_df.sort_values(by=1, ascending=False).iloc[:20, :]

    freq_count_df[0] = freq_count_df[0].apply(lambda x: x[0] + '-' + x[1])

    del flat_list, bgs, freq_count

    return freq_count_df


def bigram_graph(df, color_name, sentiment='pos'):
    

    bigram_bar_fig = px.bar(df,
                            x=1, y=0, orientation='h', text=1, color=1, color_continuous_scale=color_name,
                            labels={
                                "0": "Bi-grams",
                                "1": "Frequency",
                                "1": "Frequency"
                            },
                            height=150
                            )

    bigram_bar_fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

    bigram_bar_fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), uniformtext_minsize=4, uniformtext_mode='hide',
                                 plot_bgcolor='#070914', paper_bgcolor='#070914',
                                 font={
                                     "color": "white"
                                 })
   
    return bigram_bar_fig

def pos_area(df):
    df['date'] = df.date.apply(lambda x: x.split(' ')[0])
    df_date = df.groupby(['date', 'sentiment']).count( )

    df_date = df_date.reset_index( )

    df_date['sentiment'] = df_date['sentiment'].map({0: 'Neutral', 1: 'Positive', -1: 'Negative'})
    area_fig = px.area(df_date, x="date", y="tweet", color="sentiment", line_group="sentiment",
                       labels={
                           "tweet": "No. of tweets",
                           "date": "Dates",
                           "sentiment": "Sentiment"
                       },
                       height= 250
                       )

    area_fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), uniformtext_minsize=4, uniformtext_mode='hide',
                                 plot_bgcolor='#070914', paper_bgcolor='#070914',
                                 font={
                                     "color": "white"
                                 })

    return area_fig

def stacked_bar(df):
    df['sentiment'] = df['sentiment'].map({0: 'Neutral', -1: 'Negative', 1: 'Positive'})
    df['tweet'] = round(df['tweet']*100)
    fig = px.bar(df, x="place", y="tweet", color="sentiment", color_discrete_sequence=['#9d3f54', '#90e0ef', '#41B3A3'], opacity=0.8, text='tweet', height= 250,
                labels={
                        "place": "States",
                        "tweet": "Proportion",
                        "sentiment": "Sentiment"
                    },)

    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), uniformtext_minsize=4, uniformtext_mode='hide',
                                 plot_bgcolor='#070914', paper_bgcolor='#070914',
                                 font={
                                     "color": "white"
                                 })
    
    return fig

def motion_chart(df):
    df['date'] = df.date.apply(lambda x: x.split(' ')[0])
    df = df.groupby(['place', 'date']).count( )
    df = df.reset_index( )
 
    fig = px.scatter(df,x='date', y='tweet',animation_frame='date', 
    animation_group='place',size="tweet", 
    color='place',
    hover_name='place', log_x=True, 
    size_max=45,range_x=[200,150000], range_y=[10,100]
    )
    return fig

def mentions(df):
    df['mentions'] = df.original_tweet.apply(lambda x: list(set(re.findall('@[A-Za-z0-9_]+', x) )) if len(set(re.findall('@[A-Za-z0-9_]+', x) )) > 0 else [])
    unique_mentions = len(set(list(itertools.chain(*df['mentions'].to_list()))))
    return daq.LEDDisplay(
    
    label={'label':"Unique @mentions",
    'style': {'color': 'white'}},
    style={'padding-left': 100},
    value=str(unique_mentions),
    color="#41B3A3",
    backgroundColor='#070914',
    
)

def hashtags(df):
    df['hashtags'] = df.original_tweet.apply(lambda x: list(set(re.findall('#[A-Za-z0-9_]+', x) )) if len(set(re.findall('#[A-Za-z0-9_]+', x) )) > 0 else [])
    unique_hashtags = len(set(list(itertools.chain(*df['hashtags'].to_list()))))
    return daq.LEDDisplay(
    label={'label': "Unique #hashtags",
    'style': {'color': 'white'}},
    style={'padding-left': 40},
    value=str(unique_hashtags),
    color="#41B3A3",
    backgroundColor='#070914'
)


def boxplots(df):
    
    df['sentiment'] = df['sentiment'].map({0: 'Neutral', -1: 'Negative', 1: 'Positive'})
    df = df.replace({'place': {'Australian Capital Territory': 'ACT', 'New South Wales': 'NSW', 'Northern Territory': 'NT', 'Queensland': 'Qld', 'South Australia': 'SA', 'Tasmania': 'Tas', 'Victoria': 'Vic', 'Western Australia': 'WA'}})
    df['words'] = df['tweet'].apply(lambda x: len(x.split(' ')))

    
    fig = px.box(df, x="place", y="words", color='sentiment', color_discrete_sequence=['#90e0ef', '#41B3A3', '#9d3f54'], notched=True, height= 250,
                            labels={
                                "place": "States",
                                "words": "Word count",
                                "sentiment": "Sentiment"
                            },)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), uniformtext_minsize=4, uniformtext_mode='hide',
                                 plot_bgcolor='#070914', paper_bgcolor='#070914',
                                 font={
                                     "color": "white"
                                 })
    
    return fig
# 
# boxplots(df)
# print(px.data.tips())
pos_bigram_data = data_for_bigram(df, sentiment='pos')
neg_bigram_data = data_for_bigram(df, sentiment='neg')

max_date = df.date.max( )
asd = datetime.fromisoformat(df.date.max( )) + dt.timedelta(days=1)
today = datetime.today( )

connection.close( )

len_positive_tweets = len(positive_tweets)
len_negative_tweets = len(negative_tweets)
len_neutral_tweets = len(neutral_tweets)

tickerdata = df.original_tweet.values.tolist()[-500:]
stacked_df = df.groupby(['place', 'sentiment']).count( ) / df.groupby(['place']).count( )
stacked_df = stacked_df.drop('sentiment', axis=1).reset_index()
stacked_df = stacked_df.replace({'place': {'Australian Capital Territory': 'ACT', 'New South Wales': 'NSW', 'Northern Territory': 'NT', 'Queensland': 'Qld', 'South Australia': 'SA', 'Tasmania': 'Tas', 'Victoria': 'Vic', 'Western Australia': 'WA'}})


del positive_tweets
del negative_tweets
del neutral_tweets
# del df

app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])
# # ------------------------------------------------------------------------------



def get_div(message):
    return html.Div([message], className='ticker__item')


# # App layout
app.layout = dbc.Container([
    # First Row


    

    dbc.Row([
        
    dbc.Row([

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                        dbc.CardLink(dbc.CardImg(src='./assets/HTL_LOGO.PNG'), href='http://healthlab.edu.au/')
                ], style={'background-color':'#070914'})
                
            ], style={'height':'100%', 'width':'100%', 'display': 'flex', 'background-color':'#070914', 'overflow': 'hidden'})
        ], width=2),


        # First Row Second Col
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    
                        html.H1("Twitter Health Sentiment Analysis Dashboard (Australia YTD)",
                                style={'margin-bottom':'10px', 'margin-top':'10px', 'color': "white"},
                                className='title'
                                )
                    
                ])
            ], style={'borderRadius': '0px 25px 25px 0px',
                      'overflow': 'hidden', 'background': '#28195c'}),
            
        ],style={'margin-top': '30px', 'margin-bottom': '10px'}, width=6),

        # First Row Second Col
        dbc.Col([
            dbc.Card([

                dbc.CardBody([
                    html.Div([
                        html.Div([
                            html.Div([
                                html.Div(html.P(["Tweets", html.Br(), "relate to"]), className='pre'),
                               
                                # html.Div(html.P(["relate to"]), className='pre'),
                                html.Div([
                                    html.Div([
                                        html.Div([' medicare'], className='element'),
                                        html.Div([' health'], className='element'),
                                        html.Div([' omicron'], className='element'),
                                        html.Div([' medical'], className='element'),
                                        html.Div([' medicine'], className='element'),
                                        html.Div([' vaccine'], className='element'),
                                        html.Div([' hospital'], className='element'),
                                        html.Div([' doctor'], className='element'),
                                        html.Div([' emergency'], className='element'),
                                        html.Div([' physicians'], className='element'),
                                        html.Div([' cardiology'], className='element'),
                                        html.Div([' mammography '], className='element'),
                                        html.Div([' oncology'], className='element'),
                                        html.Div([' cancer'], className='element'),
                                        html.Div([' clinics'], className='element'),
                                        # html.Div([' primary care'], className='element'),
                                ], className='change_inner')
                            ], className='change_outer')
                        ], className='carousel')
                        ], className='center')
                    ], className='frame')
                        
                    
                ], style={'background-color': '#070914'})
            ], style={'height':'100%', 'background-color': '#070914'}),

            
        ], width=2),


        dbc.Col([
            dbc.Card([

                dbc.CardBody([
                    dbc.CardImg(src='./assets/rmit.png')
                    
                ], style={'background-color': '#070914'})
            ], style={'padding-bottom': '10px' , 'height':'100%', 'width':'100%', 'display': 'flex', 'background-color':'#070914', 'overflow': 'hidden'}),
        ], width=2)

    ], style={'background-color':'#070914'}),

    dbc.Row([

        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                               style={'borderRadius': '25px 0px 0px 0px', 'height':'15%',
                                      'overflow': 'hidden', 'background': '#E8A87C'}),

                dbc.CardBody([
                    html.H6("Positive Tweets", style={'textAlign': 'center',
                                                      'color': "black",
                                                      'background-color': '#41B3A3'}),
                    html.H2(id='content-positive', children='{}%'.format(
                        round(100 * (len_positive_tweets) / (len_positive_tweets + len_neutral_tweets + len_negative_tweets))),
                            style={'textAlign': 'center',
                                   'color': "black",
                                   'background-color': '#41B3A3'})
                ], style={'borderRadius': '0px 0px 25px 0px', 'overflow': 'hidden', 'background-color': '#41B3A3'})

            ], style={'background-color': '#070914'}, className='stats-card')
        ], width=2, style={'background-color': '#070914'}),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                               style={'borderRadius': '25px 0px 0px 0px', 'height':'15%',
                                      'overflow': 'hidden', 'background': '#E8A87C'}),

                dbc.CardBody([
                    html.H6("Negative Tweets", style={'textAlign': 'center',
                                                      'color': "black",
                                                      'background-color': '#41B3A3'}),
                    html.H2(id='content-negative', children='{}%'.format(
                        round(100 * (len_negative_tweets) / (len_positive_tweets + len_neutral_tweets + len_negative_tweets))),
                            style={'textAlign': 'center',
                                   'color': "black",
                                   'background-color': '#41B3A3'})
                ], style={'borderRadius': '0px 0px 25px 0px', 'overflow': 'hidden', 'background-color': '#41B3A3'})

            ], style={'background-color': '#070914'}, className='stats-card')
        ], width=2, style={'background-color': '#070914'}),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                               style={'borderRadius': '25px 0px 0px 0px', 'height':'15%',
                                      'overflow': 'hidden', 'background': '#E8A87C'}),

                dbc.CardBody([
                    html.H6("Neutral Tweets", style={'textAlign': 'center',
                                                      'color': "black",
                                                      'background-color': '#41B3A3'}),
                    html.H2(id='content-neutral', children='{}%'.format(
                        round(100 * (len_neutral_tweets) / (len_positive_tweets + len_neutral_tweets + len_negative_tweets))),
                            style={'textAlign': 'center',
                                   'color': "black",
                                   'background-color': '#41B3A3'})
                ], style={'borderRadius': '0px 0px 25px 0px', 'overflow': 'hidden', 'background-color': '#41B3A3'})

            ], style={'background-color': '#070914'}, className='stats-card')
        ], width=2, style={'background-color': '#070914'}),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                               style={'borderRadius': '25px 0px 0px 0px', 'height':'15%',
                                      'overflow': 'hidden', 'background': '#E8A87C'}),

                dbc.CardBody([
                    html.H6("Total Tweets Collected", style={'textAlign': 'center',
                                                             'color': "black",
                                                             'background-color': '#41B3A3'}),
                    html.H2(id='content-tweets',
                            children='{}K'.format(round((len_positive_tweets + len_negative_tweets + len_neutral_tweets) / 1000)),
                            style={'textAlign': 'center',
                                   'color': "black",
                                   'background-color': '#41B3A3'})
                ], style={'borderRadius': '0px 0px 25px 0px', 'overflow': 'hidden', 'background-color': '#41B3A3'})

            ], style={'background-color': '#070914'}, className='stats-card')
        ], width=2, style={'background-color': '#070914'}),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                               style={'borderRadius': '25px 0px 0px 0px', 'height':'15%',
                                      'overflow': 'hidden', 'background': '#E8A87C'}),

                dbc.CardBody([
                    html.H6("Total Words Analyzed", style={'textAlign': 'center',
                                                           'color': "black",
                                                           'background-color': '#41B3A3'}),
                    html.H2(id='content-words', children='{}M'.format(round(total_words/1000000,2)),
                            style={'textAlign': 'center',
                                   'color': "black",
                                   'background-color': '#41B3A3'})
                ], style={'borderRadius': '0px 0px 25px 0px', 'overflow': 'hidden', 'background-color': '#41B3A3'})

            ], style={'background-color': '#070914'}, className='stats-card')
        ], width=2, style={'background-color': '#070914'}),


       dbc.Col([
           dbc.Card([
               dbc.CardBody([
                   html.Div([html.Div([
                    html.Div([
                       html.Span(['Last updated: {} '.format(max_date.split(' ')[0])]),
             
                   ])
                  
                   ], className='text-top'),

                   html.Div([
                    html.Div([
                       html.Span(['Next update: {} '.format(asd.date( ))])
                       
                   ])
                  
                   ], className='text-bottom')], className='animated-title')


               ])
           ], style={'background-color': '#070914'})
       ])

        

    ], style={'background-color':'#070914'}, className='sec-row'),

    # Second Row
    # dbc.Row([
    #     # Second Row First Col
    #     dbc.Col([
    #         dbc.Card([
    #             dbc.CardBody([
    #                 dcc.Graph(style={'margin': '0', 'display': 'flex'},
    #                           config={"displayModeBar": False, "showTips": False},
    #                           figure=update_graph(pos_sentiment, 'Tealgrn', 'Positive tweets'))
    #             ], style={'background-color': '#070914'} )
    #         ]),
    #     ], width=6),
        
    #     dbc.Col([
    #         dbc.Card([
    #             dbc.CardBody([
    #                 dcc.Graph(style={'margin': '0'},
    #                           config={"displayModeBar": False, "showTips": False},
    #                           figure=update_graph(neg_sentiment, 'Redor', 'Negative tweets'))
    #             ],style={'background-color': '#070914'})
    #         ]),
    #     ], width=6),

    # ], style={'background-color':'#070914'}),
    ], style={'background-color': '#070914'}),
    
    dbc.Row([
        dbc.Col([
        dbc.Card([
                dbc.CardBody([
                    dcc.Graph(style={'margin': '0', 'display': 'flex'},
                              config={"displayModeBar": False, "showTips": False},
                              figure=graph(df, 'Tealgrn'))
                ], style={'background-color': '#070914'} )
            ]),  
        dbc.Card([
                


            dbc.CardHeader([
                html.H4("Trending now...",
                        style={'padding': '15px', 'borderRadius': '0px 25px 0px 0px',
                            'overflow': 'hidden', 'color': "white",
                            'background-color': '#28195c', 'height': '72px'}, className='header')
            ], style={'height': '20%'}),
            dbc.CardBody([
            html.Div([dbc.CardImg(src='./assets/pos_cloud.png', style={'margin': '0px', 'height': '400px', 'background-color': '#070914'}), 
            dbc.CardImg(src='./assets/at_cloud.png', style={'margin': '0px', 'height': '400px', 'background-color': '#070914'}) 
            ], 
            style={'padding': '0px', 'height': '50%','width': '50%', 'display': 'inline-flex', 'flex-direction': 'row'}),
        
            
        ], style={'background-color': '#070914', "outline": "solid #E8A87C", 'outline-width': 'thin'}, className='word-cloud'),



        dbc.CardBody([
                    html.Div(
                        [
                        hashtags(df), mentions(df)], style={'margin': '10px', 'padding': '0px', 'width': '20%', 'display': 'inline-flex', 'flex-direction': 'row'}
                    )
                    
                ], style={'background-color': '#070914', "outline": "solid #E8A87C", 'outline-width': 'thin'})


            ], style={'background-color': '#070914'})
    ], width=4),

    dbc.Col([
        dbc.Card([

        
        dbc.CardHeader([
            html.H4(" Top-20 Most common positive unigrams within positive tweets",
                    style={'padding': '8px', 'borderRadius': '0px 25px 0px 0px',
                            'overflow': 'hidden', 'color': "white",
                            'background-color': '#28195c', 'height': '72px'}, className='header')
        ]),
        
        dbc.CardBody([
                dcc.Graph(style={'margin':'0' , 'display': 'flex', 'width': '100%', 'height': '200%', 'font_color': 'white'},
                            config={"displayModeBar": False, "showTips": False},
                            figure=uni_gram_graph(pos_df, 'Tealgrn'))
            ], style={'margin': '0', 'background-color': '#070914', "outline": "solid #E8A87C", 'outline-width': 'thin'}),

        ], style={'background-color': '#070914'}),

        dbc.Card([

        
        dbc.CardHeader([
            html.H4("Top-20 Most common positive Bi-grams within positive tweets",
                    style={'padding': '15px', 'borderRadius': '0px 25px 0px 0px',
                            'overflow': 'hidden', 'color': "white",
                            'background-color': '#28195c', 'height': '72px'}, className='header')
        ], style={'opacity': '0.9'}),
        dbc.CardBody([
                dcc.Graph(style={'display': 'flex', 'width': '100%', 'height': '200%', 'font_color': 'white'},
                            config={"displayModeBar": False, "showTips": False},
                            figure=bigram_graph(pos_bigram_data, 'Tealgrn', 'pos'))
            ], style={'background-color': '#070914', "outline": "solid #E8A87C", 'outline-width': 'thin'})

        ], style={'background-color': '#070914'}),

        dbc.Card([
            dbc.CardHeader([
            html.H4("Tweets collected over time",
                    style={'padding': '15px', 'borderRadius': '0px 25px 0px 0px',
                        'overflow': 'hidden', 'color': "white",
                        'background-color': '#28195c', 'height': '72px'}, className='header')
        ], style={'opacity': '0.9'}),
                dbc.CardBody([
                    dcc.Graph(style={'margin': '0px', 'width': '100%', 'font_color': 'white'},
                        config={"displayModeBar": False, "showTips": False}, figure=pos_area(df))
                ], style={'background-color': '#070914', "outline": "solid #E8A87C", 'outline-width': 'thin'})

                # dbc.CardBody([
                #     dcc.Graph(style={'margin': '0px', 'width': '100%', 'font_color': 'white'},
                #         config={"displayModeBar": False, "showTips": False}, figure=motion_chart(df))
                # ], style={'background-color': '#070914', "outline": "solid #E8A87C", 'outline-width': 'thin'})

                # dbc.CardBody([
                #     html.Div(
                #         [mentions(df),
                #         hashtags(df)], style={'padding': '5px', 'width': '20%', 'display': 'inline-flex', 'flex-direction': 'row'}
                #     )
                    
                # ], style={'background-color': '#070914', "outline": "solid #E8A87C", 'outline-width': 'thin'})

                # dbc.CardBody([
                #     dcc.Graph(style={'margin': '0px', 'width': '100%', 'font_color': 'white'},
                #         config={"displayModeBar": False, "showTips": False}, figure=boxplots(df))
                # ], style={'background-color': '#070914', "outline": "solid #E8A87C", 'outline-width': 'thin'})
                
            ], style={'background-color': 'rgb(0,0,0)'}, className='area-graph')


    ], width=4),

    dbc.Col([
        dbc.Card([
        dbc.CardHeader([
        html.H4("Top-20 Most common negative unigrams within negative tweets",
                style={'padding': '8px', 'borderRadius': '0px 25px 0px 0px',
                        'overflow': 'hidden', 'color': "white",
                        'background-color': '#28195c', 'height': '72px'}, className='header')
        ]),

        dbc.CardBody([
            dcc.Graph(style={'margin':'0', 'display': 'flex', 'width': '100%', 'height': '200%', 'font_color': 'white'},
                        config={"displayModeBar": False, "showTips": False},
                        figure=uni_gram_graph(neg_df, 'Redor'))
        ], style={'margin': '0', 'background-color': '#070914', "outline": "solid #E8A87C", 'outline-width': 'thin'}),
        ], style={'background-color': '#070914'}),

        dbc.Card([
        dbc.CardHeader([
        html.H4(" Top-20 Most common negative Bi-grams within negative tweets",
                style={'padding': '15px', 'borderRadius': '0px 25px 0px 0px',
                        'overflow': 'hidden', 'color': "white",
                        'background-color': '#28195c', 'height': '72px'}, className='header')
    ]),
    dbc.CardBody([
            dcc.Graph(style={'display': 'flex', 'width': '100%', 'height': '200%', 'font_color': 'white'},
                        config={"displayModeBar": False, "showTips": False},
                        figure=bigram_graph(neg_bigram_data, 'Redor', 'neg'))
        ], style={'background-color': '#070914', "outline": "solid #E8A87C", 'outline-width': 'thin'}), 
        ], style={'background-color': '#070914'}),

        dbc.Card([
        dbc.CardHeader([
        html.H4(" Sentiment composition by states ",
                style={'padding': '15px', 'borderRadius': '0px 25px 0px 0px',
                        'overflow': 'hidden', 'color': "white",
                        'background-color': '#28195c', 'height': '72px'}, className='header')
    ]),
    dbc.CardBody([
            dcc.Graph(style={'display': 'flex', 'width': '100%', 'height': '200%', 'font_color': 'white'},
                        config={"displayModeBar": False, "showTips": False},
                        figure=stacked_bar(stacked_df))
        ], style={'background-color': '#070914', "outline": "solid #E8A87C", 'outline-width': 'thin'}), 
        ], style={'background-color': '#070914'}),

    
    ], style={'width': 4}),

    
    
    
    ], style={'display': 'flex', 'background-color': '#070914'}),
        
    dbc.Row([
        dbc.Card([

        ], style={'background-color': '#070914'})
    ], style={'display': 'flex', 'background-color': '#070914'}, className='content-dash'),
    

#     # Graph headlines (unigrams)
#    dbc.Row([
#        dbc.Col([

#         dbc.Card([

        
#         dbc.CardHeader([
#             html.H4(" Top-20 Most common positive unigrams within positive tweets",
#                     style={'padding': '8px', 'borderRadius': '0px 25px 0px 0px',
#                             'overflow': 'hidden', 'color': "white",
#                             'background-color': '#28195c', 'height': '72px'}, className='header')
#         ]),
        
#         dbc.CardBody([
#                 dcc.Graph(style={'margin':'0' , 'display': 'flex', 'width': '100%', 'height': '200%', 'font_color': 'white'},
#                             config={"displayModeBar": False, "showTips": False},
#                             figure=uni_gram_graph(pos_df, 'Tealgrn'))
#             ], style={'margin': '0', 'background-color': '#070914', "outline": "solid #E8A87C", 'outline-width': 'thin'}),

#         ], style={'background-color': '#070914'}),


#         dbc.CardHeader([
#             html.H4("Top-20 Most common positive Bi-grams within positive tweets",
#                     style={'padding': '15px', 'borderRadius': '0px 25px 0px 0px',
#                             'overflow': 'hidden', 'color': "white",
#                             'background-color': '#28195c', 'height': '72px'}, className='header')
#         ], style={'opacity': '0.9'}),
#         dbc.CardBody([
#                 dcc.Graph(style={'display': 'flex', 'width': '100%', 'height': '200%', 'font_color': 'white'},
#                             config={"displayModeBar": False, "showTips": False},
#                             figure=bigram_graph(pos_bigram_data, 'Tealgrn', 'pos'))
#             ], style={'background-color': '#070914', "outline": "solid #E8A87C", 'outline-width': 'thin'})

#     ], style={'height': "10%"}, width=4),


    
    # dbc.Col([
    #         dbc.Card([
                
    #                 dbc.CardHeader([
    #                     html.H4("What are people talking about? (Healthcare)",
    #                             style={'padding': '15px', 'borderRadius': '0px 25px 0px 0px',
    #                                 'overflow': 'hidden', 'color': "white",
    #                                 'background-color': '#28195c', 'height': '72px'}, className='header')
    #                 ]),
    #                 dbc.CardBody([
    #                 dbc.CardImg(src='./assets/pos_cloud.png')
    #             ], style={'background-color': '#070914', "outline": "solid #E8A87C", 'outline-width': 'thin'})


    #         ], style={'background-color': '#070914'})

    #     ], width=4),

    

#     dbc.Col([
#     dbc.CardHeader([
#         html.H4("Top-20 Most common negative unigrams within negative tweets",
#                 style={'padding': '8px', 'borderRadius': '0px 25px 0px 0px',
#                         'overflow': 'hidden', 'color': "white",
#                         'background-color': '#28195c', 'height': '72px'}, className='header')
#     ]),

#     dbc.CardBody([
#             dcc.Graph(style={'display': 'flex', 'width': '100%', 'height': '200%', 'font_color': 'white'},
#                         config={"displayModeBar": False, "showTips": False},
#                         figure=uni_gram_graph(neg_df, 'Redor'))
#         ], style={'background-color': '#070914', "outline": "solid #E8A87C", 'outline-width': 'thin'}),

    

#     dbc.CardHeader([
#         html.H4(" Top-20 Most common negative Bi-grams within negative tweets",
#                 style={'padding': '15px', 'borderRadius': '0px 25px 0px 0px',
#                         'overflow': 'hidden', 'color': "white",
#                         'background-color': '#28195c', 'height': '72px'}, className='header')
#     ]),
#     dbc.CardBody([
#             dcc.Graph(style={'display': 'flex', 'width': '100%', 'height': '200%', 'font_color': 'white'},
#                         config={"displayModeBar": False, "showTips": False},
#                         figure=bigram_graph(neg_bigram_data, 'Redor', 'neg'))
#         ], style={'background-color': '#070914', "outline": "solid #E8A87C", 'outline-width': 'thin'}), 

    

#     ],style={'height': "10%"}, width=4),
#    ], style={'background-color':'#070914'}),

dbc.Row([
    dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Div([

                    get_div(i)  for i in tickerdata
                    
                ], className='ticker')
            ], style={'background-color': '#28195c', 'margin': '0px'}, className='ticker-wrap'),
        ],style={'background-color': '#28195c', 'margin': '0px'})
    ],style={'background-color': '#28195c', 'margin': '0px'})
], style={'display':'flex', 'background-color': '#28195c', 'margin': '0px'})


   
], style={'background-color': 'rgb(0,0,0)'}, fluid=True)

if __name__ == '__main__':

    app.run_server(host='0.0.0.0', debug=True, port=8050)

