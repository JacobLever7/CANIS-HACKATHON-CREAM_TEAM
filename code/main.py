# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Load the dataset
df = pd.read_csv('./data/archive/DataSet_Misinfo_FAKE.csv')

# Data Cleaning
df.drop_duplicates(inplace=True)
df.dropna(subset=['title', 'text'], inplace=True)
df.drop(['uuid', 'ord_in_thread', 'published', 'language'], axis=1, inplace=True)

# Exploratory Data Analysis
topics = df['thread_title'].value_counts().nlargest(10)
plt.figure(figsize=(10,5))
sns.barplot(x=topics.values, y=topics.index, palette='Blues_r')
plt.title('Most Common Topics of Misinformation and Fake News')
plt.xlabel('Number of Articles')
plt.ylabel('Topic')
plt.show()

sources = df['site_url'].value_counts().nlargest(10)
plt.figure(figsize=(10,5))
sns.barplot(x=sources.values, y=sources.index, palette='Greens_r')
plt.title('Most Common Sources of Misinformation and Fake News')
plt.xlabel('Number of Articles')
plt.ylabel('Source')
plt.show()

# Feature Engineering
stop_words = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(stop_words=stop_words)
X = vectorizer.fit_transform(df['text'])
df['title_sentiment'] = df['title'].apply(lambda x: SentimentIntensityAnalyzer().polarity_scores(x)['compound'])
df['text_sentiment'] = df['text'].apply(lambda x: SentimentIntensityAnalyzer().polarity_scores(x)['compound'])
df['num_words'] = df['text'].apply(lambda x: len(x.split()))

# Machine Learning Modeling
X_train, X_test, y_train, y_test = train_test_split(X, df['type'], test_size=0.3, random_state=42)

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
print('Logistic Regression Results:')
print('Accuracy:', accuracy_score(y_test, lr_preds))
print('Precision:', precision_score(y_test, lr_preds))
print('Recall:', recall_score(y_test, lr_preds))
print('F1 Score:', f1_score(y_test, lr_preds))
print('Confusion Matrix:')
print(confusion_matrix(y_test, lr_preds))

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
print('\nRandom Forest Results:')
print('Accuracy:', accuracy_score(y_test, rf_preds))
print('Precision:', precision_score(y_test, rf_preds))
print('Recall:', recall_score(y_test, rf_preds))
print('F1 Score:', f1_score(y_test, rf_preds))
print('Confusion Matrix:')
print(confusion_matrix(y_test, rf_preds))

# Grid Search Cross Validation
param_grid = {'n_estimators': [50, 100, 150],
              'max_depth': [10, 20, 30, None]}
rf_gs = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
rf_gs.fit(X_train, y_train)
rf_gs_preds = rf_gs.predict(X_test)
print('\nRandom Forest with Grid Search Results:')
print('Best Parameters:', rf_gs.best_params_)
print('Accuracy:', accuracy_score(y_test, rf_gs_preds))
print('Precision:', precision_score(y_test, rf_gs_preds))
print('Recall:', recall_score(y_test, rf_gs_preds))
print('F1 Score:', f1_score(y_test, rf_gs_preds))
print('Confusion Matrix:')
print(confusion_matrix(y_test, rf_gs_preds))