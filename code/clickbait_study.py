import pandas as pd
import os
import matplotlib.pyplot as plt
import string
import re
from textblob import TextBlob
from functions import *


df_fake_raw, df_real_raw = load_datasets()

df_fake, df_real = cleanup_datasets(df_fake_raw, df_real_raw)


# define a function to detect clickbait language in a text string
def detect_clickbait(text):
    blob = TextBlob(text.lower())
    clickbait_words = ['you won\'t believe', 'shocking', 'mind-blowing', 'outrageous', 'jaw-dropping', 'unbelievable', 'amazing', 'stunning']
    count = sum([1 for word in clickbait_words if word in blob])
    return count

# apply the detect_clickbait function to the 'text' column of each dataframe
df_real['clickbait'] = df_real['text'].apply(detect_clickbait)
df_fake['clickbait'] = df_fake['text'].apply(detect_clickbait)

# calculate the frequency of clickbait language in each dataframe
real_freq = df_real['clickbait'].sum() / len(df_real)
fake_freq = df_fake['clickbait'].sum() / len(df_fake)

# create a bar chart of the clickbait frequency for each dataframe
plt.bar(['Real', 'Fake'], [real_freq, fake_freq])
plt.xlabel('News Type')
plt.ylabel('Frequency of Clickbait Language')
plt.title('Clickbait Language in Real and Fake News')
plt.show()