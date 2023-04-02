
import pandas as pd
import os
import matplotlib.pyplot as plt
import string
import re
from textblob import TextBlob
from functions import *


df_fake_raw, df_real_raw = load_datasets()

df_fake, df_real = cleanup_datasets(df_fake_raw, df_real_raw)


# define a function to detect emotional language
def detect_emotion(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return 'positive'
    elif sentiment < 0:
        return 'negative'
    else:
        return 'neutral'

# apply the detect_emotion function to the 'text' column of each dataframe
df_fake['emotion'] = df_fake['text'].apply(detect_emotion)
df_real['emotion'] = df_real['text'].apply(detect_emotion)

# create a list of emotional categories
emotions = ['positive', 'neutral', 'negative']

# create a dictionary to hold the frequency of each emotional category for each dataframe
freq_dict = {'Fake': [], 'Real': []}
for emo in emotions:
    freq_dict['Fake'].append(df_fake['emotion'].value_counts()[emo])
    freq_dict['Real'].append(df_real['emotion'].value_counts()[emo])

# create a bar chart of the emotional frequency for each dataframe
bar_width = 0.35
r1 = range(len(emotions))
r2 = [x + bar_width for x in r1]
plt.bar(r1, freq_dict['Fake'], color='red', width=bar_width, label='Fake')
plt.bar(r2, freq_dict['Real'], color='green', width=bar_width, label='Real')
plt.xticks([r + bar_width/2 for r in r1], emotions)
plt.xlabel('Emotion')
plt.ylabel('Frequency')
plt.title('Emotional Frequency in Fake and Real News')
plt.legend()
plt.show()