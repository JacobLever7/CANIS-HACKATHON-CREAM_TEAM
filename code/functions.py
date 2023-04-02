import pandas as pd
import os
import string
import re
from tqdm import tqdm
import re
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download(['punkt', 'stopwords'])

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer


def load_datasets():
    # Load the CSV files
    path = os.path.realpath(__file__)
    dir = os.path.dirname(path)
    dir = dir.replace('code', 'data')
    os.chdir(dir)

    df_fake = pd.read_csv('archive/DataSet_Misinfo_FAKE.csv')
    df_real = pd.read_csv('archive/DataSet_Misinfo_TRUE.csv')

    return df_fake, df_real

def cleanup_datasets(df_fake, df_real):
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

    # Convert text to type str
    df_real['text'] = df_real['text'].astype(str)
    df_fake['text'] = df_fake['text'].astype(str)

    return df_fake, df_real


def preprocess_text(text_data):
    preprocessed_text = []
      
    for sentence in tqdm(text_data):
        sentence = re.sub(r'[^\w\s]', '', sentence)
        preprocessed_text.append(' '.join(token.lower()
                                  for token in str(sentence).split()
                                  if token not in stopwords.words('english')))
  
    return preprocessed_text

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx])
                  for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1],
                        reverse=True)
    return words_freq[:n]
