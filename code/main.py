import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob

# load the fake and real dataframes
import pandas as pd
import os
import matplotlib.pyplot as plt
from textblob import TextBlob

# Load the dataset
path = os.path.realpath(__file__)
dir = os.path.dirname(path)
dir = dir.replace('code', 'data')
os.chdir(dir)

df_fake = pd.read_csv('archive/DataSet_Misinfo_FAKE.csv')
df_real = pd.read_csv('archive/DataSet_Misinfo_TRUE.csv')

def detect_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    return sentiment

# apply the detect_sentiment function to the 'text' column of each dataframe
df_fake['sentiment_score'] = df_fake['text'].apply(detect_sentiment)
df_real['sentiment_score'] = df_real['text'].apply(detect_sentiment)

# assign a sentiment label based on the sentiment score
df_fake.loc[df_fake['sentiment_score'] > 0, 'sentiment_label'] = 'positive'
df_fake.loc[df_fake['sentiment_score'] == 0, 'sentiment_label'] = 'neutral'
df_fake.loc[df_fake['sentiment_score'] < 0, 'sentiment_label'] = 'negative'

df_real.loc[df_real['sentiment_score'] > 0, 'sentiment_label'] = 'positive'
df_real.loc[df_real['sentiment_score'] == 0, 'sentiment_label'] = 'neutral'
df_real.loc[df_real['sentiment_score'] < 0, 'sentiment_label'] = 'negative'

# create a list of sentiment categories
sentiments = ['positive', 'neutral', 'negative']

# create a dictionary to hold the frequency of each sentiment category for each dataframe
freq_dict = {'Fake': [], 'Real': []}
for sent in sentiments:
    freq_dict['Fake'].append(df_fake['sentiment_label'].value_counts()[sent])
    freq_dict['Real'].append(df_real['sentiment_label'].value_counts()[sent])

# create a bar chart of the sentiment frequency for each dataframe
bar_width = 0.35
r1 = range(len(sentiments))
r2 = [x + bar_width for x in r1]
plt.bar(r1, freq_dict['Fake'], color='red', width=bar_width, label='Fake')
plt.bar(r2, freq_dict['Real'], color='green', width=bar_width, label='Real')
plt.xticks([r + bar_width/2 for r in r1], sentiments)
plt.xlabel('Sentiment')
plt.ylabel('Frequency')
plt.title('Sentiment Frequency in Fake and Real News')
plt.legend()
plt.show()
