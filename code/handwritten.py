
import pandas as pd
import os
import matplotlib.pyplot as plt



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

# Combine the dataframes
df_combined = pd.concat([df_fake, df_real], ignore_index=True)

# Calculate summary statistics
num_fake_articles = len(df_fake)
num_real_articles = len(df_real)

# Set the colors for the bars
colors = ['red', 'green']

# Create the bar chart
plt.bar(['Fake', 'Real'], [num_fake_articles, num_real_articles], color=colors)

# Add text above each bar to show the exact number of articles
plt.text(0, num_fake_articles + 1000, num_fake_articles, ha='center')
plt.text(1, num_real_articles + 1000, num_real_articles, ha='center')

# Create the bar chart
plt.title('Number of articles in each category')
plt.xlabel('Category')
plt.ylabel('Number of articles')
plt.ylim(0, max(num_fake_articles, num_real_articles) * 1.2)
plt.show()