#this is where we will put our algorithm to detect fake news.

#import dependancies
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from functions import *


df_fake_raw, df_real_raw = load_datasets()

df_fake, df_real = cleanup_datasets(df_fake_raw, df_real_raw)

# Combine the dataframes
data = pd.concat([df_fake, df_real], ignore_index=True)

# Shuffling(randomizes the dataset to minimize bias)
data = data.sample(frac=1)
data.reset_index(inplace=True)
data.drop(["index"], axis=1, inplace=True)


# Preprocess data 
preprocessed_review = preprocess_text(data['text'].values)
data['text'] = preprocessed_review

# # Real
# consolidated = ' '.join(
#     word for word in data['text'][data['label'] == 1].astype(str))
# wordCloud = WordCloud(width=1600,
#                       height=800,
#                       random_state=21,
#                       max_font_size=110,
#                       collocations=False)
# plt.figure(figsize=(15, 10))
# plt.imshow(wordCloud.generate(consolidated), interpolation='bilinear')
# plt.axis('off')


common_words = get_top_n_words(data['text'], 20)
df1 = pd.DataFrame(common_words, columns=['Review', 'count'])
  
df1.groupby('Review').sum()['count'].sort_values(ascending=False).plot(
    kind='bar',
    figsize=(10, 6),
    xlabel="Top Words",
    ylabel="Count",
    title="Bar Chart of Top Words Frequency"
)
plt.show()


# Split into train and test
x_train, x_test, y_train, y_test = train_test_split(data['text'], 
                                                    data['label'], 
                                                    test_size=0.25)

# Convert into vectors using TfidfVectorizer
vectorization = TfidfVectorizer()
x_train = vectorization.fit_transform(x_train)
x_test = vectorization.transform(x_test) 

# Train using logistic regression
model = LogisticRegression()
model.fit(x_train, y_train)
  
## testing the model
print(accuracy_score(y_train, model.predict(x_train)))
print(accuracy_score(y_test, model.predict(x_test)))


# Train using Decision Tree classifier
model = DecisionTreeClassifier()
model.fit(x_train, y_train)
  
## testing the model
print(accuracy_score(y_train, model.predict(x_train)))
print(accuracy_score(y_test, model.predict(x_test)))

# Confusion matrix of results from Decision Tree
cm = metrics.confusion_matrix(y_test, model.predict(x_test))
  
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                                            display_labels=[False, True])
  
cm_display.plot()
plt.show()