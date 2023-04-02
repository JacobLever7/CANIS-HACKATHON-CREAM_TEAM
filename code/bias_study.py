from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import string
from functions import *


df_fake_raw, df_real_raw = load_datasets()

df_fake, df_real = cleanup_datasets(df_fake_raw, df_real_raw)

# define function to detect bias language
def detect_bias(text):
    blob = TextBlob(text)
    count = 0
    for sentence in blob.sentences:
        if 'they' in sentence or 'them' in sentence or 'their' in sentence:
            count += 1
        if 'we' in sentence or 'us' in sentence or 'our' in sentence:
            count -= 1
    if count > 0:
        return 'biased against them'
    elif count < 0:
        return 'biased against us'
    else:
        return 'neutral'

# apply the detect_bias function to the 'text' column of each dataframe
df_fake['bias'] = df_fake['text'].apply(detect_bias)
df_real['bias'] = df_real['text'].apply(detect_bias)

# create a list of bias categories
bias_categories = ['biased against us', 'neutral', 'biased against them']

# create a dictionary to hold the frequency of each bias category for each dataframe
freq_dict = {'Fake': [], 'Real': []}
for bias in bias_categories:
    freq_dict['Fake'].append(df_fake['bias'].value_counts()[bias])
    freq_dict['Real'].append(df_real['bias'].value_counts()[bias])

# create a bar chart of the bias frequency for each dataframe
bar_width = 0.35
r1 = range(len(bias_categories))
r2 = [x + bar_width for x in r1]
plt.bar(r1, freq_dict['Fake'], color='red', width=bar_width, label='Fake')
plt.bar(r2, freq_dict['Real'], color='green', width=bar_width, label='Real')
plt.xticks([r + bar_width/2 for r in r1], bias_categories)
plt.xlabel('Bias Language')
plt.ylabel('Frequency')
plt.title('Bias Language Frequency in Fake and Real News')
plt.legend()
plt.show()