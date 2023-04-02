
import pandas as pd
import os
import matplotlib.pyplot as plt
import string
import re



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


# Combine the dataframes
df_combined = pd.concat([df_fake, df_real], ignore_index=True)

# Calculate summary statistics
num_fake_articles = len(df_fake)
num_real_articles = len(df_real)

# # Set the colors for the bars
# colors = ['red', 'green']

# # Create the bar chart
# plt.bar(['Fake', 'Real'], [num_fake_articles, num_real_articles], color=colors)

# # Add text above each bar to show the exact number of articles
# plt.text(0, num_fake_articles + 1000, num_fake_articles, ha='center')
# plt.text(1, num_real_articles + 1000, num_real_articles, ha='center')

# # Create the bar chart
# plt.title('Number of articles in each category')
# plt.xlabel('Category')
# plt.ylabel('Number of articles')
# plt.ylim(0, max(num_fake_articles, num_real_articles) * 1.2)
# plt.show()

# Character Length of Titles - Min, Mean, Max for df_fake
print('df_fake')
print('Mean Length:', df_fake['text'].apply(len).mean())
print('Min Length:', df_fake['text'].apply(len).min())
print('Max Length:', df_fake['text'].apply(len).max())

# Character Length of Titles - Min, Mean, Max for df_real
print('\ndf_real')
print('Mean Length:', df_real['text'].apply(len).mean())
print('Min Length:', df_real['text'].apply(len).min())
print('Max Length:', df_real['text'].apply(len).max())

# Plot histograms for both df_fake and df_real
plt.hist(df_fake['text'].apply(len), bins=335, alpha=0.5, color='red', label='df_fake')
plt.hist(df_real['text'].apply(len), bins=200, alpha=0.5, color='blue', label='df_real')
plt.legend(loc='upper right')
plt.title('Distribution of Text Length')
plt.xlabel('Text Length')
plt.ylabel('Frequency')
# Remove far-right outliers
plt.xlim(left=-2500, right=20000)
plt.show()