from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import string

# Load the CSV files
path = os.path.realpath(__file__)
dir = os.path.dirname(path)
dir = dir.replace('code', 'data')
os.chdir(dir)

df_fake = pd.read_csv('archive/DataSet_Misinfo_FAKE.csv')
df_real = pd.read_csv('archive/DataSet_Misinfo_TRUE.csv')


# Add label column to each dataframe
df_fake['label'] = 0
df_real['label'] = 1

#fix weird column name: 'Unnamed: 0'
df_fake.rename(columns= {'Unnamed: 0' : 'title'}, inplace=True)
df_real.rename(columns= {'Unnamed: 0' : 'title'}, inplace=True)

# Drop the title column
df_fake.drop(columns=["title"], inplace=True)
df_real.drop(columns=["title"], inplace=True)

# clean up data
df_fake['text'] = df_fake['text'].astype(str)
df_real['text'] = df_real['text'].astype(str)

# lowercase
df_fake['text'] = df_fake['text'].str.lower()
df_real['text'] = df_real['text'].str.lower()

# remove punctuation
df_fake['text'] = df_fake['text'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
df_real['text'] = df_real['text'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

# Remove URLs and HTML tags
df_fake['text'] = df_fake['text'].apply(lambda x: re.sub(r'http\S+', '', x))
df_fake['text'] = df_fake['text'].apply(lambda x: re.sub(r'<.*?>', '', x))

df_real['text'] = df_real['text'].apply(lambda x: re.sub(r'http\S+', '', x))
df_real['text'] = df_real['text'].apply(lambda x: re.sub(r'<.*?>', '', x))

df_real['text'] = df_real['text'].astype(str)
df_fake['text'] = df_fake['text'].astype(str)

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