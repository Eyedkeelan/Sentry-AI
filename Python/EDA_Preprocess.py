import os
import cv2 as cv
import numpy as np
import pandas as pd 
import torch
from torch import nn
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from matplotlib import pyplot as plt
import re

from tqdm import tqdm


BASE_DIR = r"C:\Users\Keelan.Butler\Desktop\python_projects\Final Project"

ANOM_PATH = os.path.join(BASE_DIR, "OneDrive_2025-01-30", "MSAD Dataset", "anomaly_annotation.csv")
META_PATH = os.path.join(BASE_DIR, "Dataset", "metadata.csv")
TEST_PATH = os.path.join(BASE_DIR, "Dataset", "MSAD_I3D_WS_Test.list")
TRAIN_PATH = os.path.join(BASE_DIR, "Dataset", "MSAD_I3D_WS_Train.list")



ANOM_df = pd.read_csv(ANOM_PATH)

META_df = pd.read_csv(META_PATH)
print(f"{ANOM_df.head(3)}\n{META_df.head(3)}")


META_df['name'] = [row[:-4] for row in META_df.video_file.values] # This will create an ID in which we may merge tables on. 
print(META_df.name)
df = META_df.merge(ANOM_df,how='left',on='name')


df['starting frame of anomaly'].fillna(99999,inplace= True)
df['ending frame of anomaly'].fillna(99999,inplace= True) # I have set these frames to a high value as this enables all 'normal' events to not be registered within the anomaly ranges

df['Anomaly'] = np.where((df['starting frame of anomaly'] <= df['frame_number']) & 
                         (df['ending frame of anomaly'] >= df['frame_number']), 1, 0)

df['Anomaly_Type'] = np.where((df['starting frame of anomaly'] <= df['frame_number']) & 
                              (df['ending frame of anomaly'] >= df['frame_number']),
                              df['name'].str.split('_').str[0], "Normal")
corrections = {
    "MSAD_normal_testing_": np.nan,
    "perdestrian_street": "pedestrian_street"
            }
df['scenario'] = df['scenario'].fillna(df['name'].apply(lambda x: re.sub(r'\d+', '', x))) # Due to alot of normal data not containing scenarios, these can be extracted from the name
df['scenario'] = df['scenario'].replace(corrections) # Removing MSAD normal testing at not relevant, corrected spelling mistakes.
df['name'] = df['name'].replace(corrections)
# df.to_csv("joined_table.csv")

print(df.info(),df.describe())
# Exploration of scenario and anomaly, where 1 represents anomaly occuring. VISULISATIONS.
sns.countplot(df,x = "scenario", hue = 'Anomaly')
plt.title("Frames by Situation")
plt.xlabel('Scenario',fontdict={'size':10,'weight':'bold'})
plt.xticks(rotation = 'vertical')
plt.ylabel('Frames',fontdict={'size':10,'weight':'bold'})
plt.grid()
plt.show()
# HEAT MAP WITH SPECIFIC NUMERICALS
pivot_table = df.pivot_table(index='scenario', columns='Anomaly', aggfunc='size', fill_value=0)
sns.heatmap(pivot_table, annot=True, fmt='d', cmap='coolwarm')
plt.title('Anomaly Counts by Scenario')
plt.show()
# Class balance
sns.countplot(x='Anomaly', data=df)
plt.title('Anomaly vs Normal Distribution')
plt.show()
print(df['Anomaly'].value_counts(normalize=True) * 100)

# We will follow the predefined test-train split
with open(TRAIN_PATH, 'r') as train, open(TEST_PATH,'r') as test:
    train_list, test_list = train.readlines(), test.readlines()

def process_list(l):
    final_list = []
    for i in l: 
        i0 = i.split(r'/')[-1]
        i0 = i0.rpartition('_')[0]
        if 'MSAD_normal_' in i0: 
            i0 = i0.split('MSAD_normal_')[1]
        final_list.append(i0)
    return final_list

train_list = process_list(train_list)
test_list = process_list(test_list)

print(f'Training set: {train_list}\n Training size: {len(train_list)}\n' )
print(f'Testing set: {test_list}\n{len(test_list)}\n')

train_df_appender = pd.DataFrame(train_list, columns=['name'])
test_df_appender = pd.DataFrame(test_list, columns=['name'])


train_df_appender['Partition'] = 'Train'
test_df_appender['Partition'] = 'Test'


df_appender = pd.concat([train_df_appender, test_df_appender])
df_appender['name'] = df_appender['name'].replace({"MSAD_normal_testing_":'', "perdestrian_street":"pedestrian_street"})
df_appender['name'] = df_appender['name'].replace('perdestrian', 'pedestrian', regex=True)
df['name'] = df['name'].replace('perdestrian', 'pedestrian', regex=True)
df['name'] = df['name'].replace('MSAD_normal_', '', regex=True)
df = df.merge(df_appender, how='left', on='name')

df.to_csv('test_train.csv')

df_train = df[df['Partition'] == 'Train'].drop(columns=['Partition'])
df_test = df[df['Partition'] == 'Test'].drop(columns=['Partition']) 

# We will check for data leakage, this will ensure a true reflection on the generalisation of model
overlap = set(df_train['name']).intersection(set(df_test['name']))
print(f"Overlap between train and test sets: {overlap}")


#T-sne, first I will extract CNN features and corresponding labels (type of anomaly)

features = []
anon_type = []
anon_bool = []

for path, label, lab_bool in tqdm(zip(df_train['feature_path'],df_train['Anomaly_Type'],df_train['Anomaly'])):
    feature = np.load(path)
    features.append(feature)
    anon_type.append(label)
    anon_bool.append(lab_bool)

features = np.array(features)
anon_type = np.array(anon_type)
anon_bool = np.array(anon_bool)

# Standardise features

scaler = StandardScaler() 
features_scaled = scaler.fit_transform(features)
df_train.to_csv('Train_set.csv')
df_test.to_csv('Test_set.csv')
# Applying to the t-sne with perplexity optimisation\

kldiv_list = []
perplex_list = [5, 15 , 30, 40 , 50, 60, 100]
for perplexity in tqdm(perplex_list):
    tsne = TSNE(n_components=2, perplexity = perplexity ,n_iter=1000, random_state= 42)
    tsne.fit_transform(features_scaled)
    kldiv_list.append(tsne.kl_divergence_)

sns.lineplot(x = perplex_list, y = kldiv_list)

plt.xlabel("Perplexity")
plt.ylabel("Kullback–Leibler divergence")
plt.title("T-distributed Stochastic Neighbor Embedding (Perplexity fine-tuning)")
plt.show()


tsne = TSNE(n_components=2, perplexity= 50,n_iter=2500, random_state= 42)
tsne_results = tsne.fit_transform(features_scaled)

# Visualisation

plt.figure(figsize=(10, 8))
sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=anon_bool, palette=['blue', 'red'], alpha=0.7)
plt.title('t-SNE Anomaly Visualization of Image Features')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend(title='Anomaly')
plt.show()

plt.figure(figsize=(10, 8))
sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=anon_type, alpha=0.7)
plt.title('t-SNE by Anomaly Type Visualization of Image Features')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend(title='Anomaly')
plt.show()

