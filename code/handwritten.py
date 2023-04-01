
import pandas as pd
import os


# Load the dataset
path = os.path.realpath(__file__)
dir = os.path.dirname(path)
dir = dir.replace('code', 'data')
os.chdir(dir)

df_fake = pd.read_csv('archive/DataSet_Misinfo_FAKE.csv')
df_real = pd.read_csv('archive/DataSet_Misinfo_TRUE.csv')


#fix weird column name: 'Unnamed: 0'
df_fake.rename(columns= {'Unnamed: 0' : 'title'}, inplace=True)
df_real.rename(columns= {'Unnamed: 0' : 'title'}, inplace=True)

# Add label column to each dataframe
label_dict = {'df_fake': 0, 'df_real': 1}
df_fake['label'] = df_fake['title'].map(lambda x: label_dict['df_fake'])
df_real['label'] = df_real['title'].map(lambda x: label_dict['df_real'])

# Drop the title column
df_fake.drop(columns=["title"], inplace=True)
df_real.drop(columns=["title"], inplace=True)

# Combine the dataframes and create multi-index dataframe
df_combined = pd.concat([df_fake, df_real], keys=['fake', 'real'], ignore_index=True)

print(df_combined)

