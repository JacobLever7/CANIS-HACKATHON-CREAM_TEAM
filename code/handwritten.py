import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

#add label to signify fake:0 or real:1 data
for item in df_fake:
    df_fake['label'] = 0

for item in df_real:
    df_real['label'] = 1

#strip data for combination
df_real.drop(columns=["title"], inplace=True)
df_fake.drop(columns=["title"], inplace=True)

df_combined = pd.concat([df_real, df_fake], ignore_index=True)

print(df_combined)

