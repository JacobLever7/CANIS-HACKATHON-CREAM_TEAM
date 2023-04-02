import pandas as pd
import os
import matplotlib.pyplot as plt
import string
import re
from textblob import TextBlob


# Load the dataset
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