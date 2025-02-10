import os
import cv2 as cv
import numpy as np
import pandas as pd 
import torch
from torch import nn
import seaborn as sns
from matplotlib import pyplot as plt
import re

from tqdm import tqdm


PATH_DIR = r"C:\Users\Keelan.Butler\Desktop\python_projects\Final Project\Dataset\features\MSAD_anomaly_blur\Assault\Assault_1.mp4_frame_0.npy" 

ANOM_PATH = r"C:\Users\Keelan.Butler\Desktop\python_projects\Final Project\OneDrive_2025-01-30\MSAD Dataset\anomaly_annotation.csv"

META_PATH = r"C:\Users\Keelan.Butler\Desktop\python_projects\Final Project\Dataset\metadata.csv"



ANOM_df = pd.read_csv(ANOM_PATH)

META_df = pd.read_csv(META_PATH)
print(f"{ANOM_df.head(3)}\n{META_df.head(3)}")


META_df['name'] = [row[:-4] for row in META_df.video_file.values] # This will create an ID in which we may merge tables on. 
print(META_df.name)
df = META_df.merge(ANOM_df,how='left',on='name')


df['starting frame of anomaly'].fillna(99999,inplace= True)
df['ending frame of anomaly'].fillna(99999,inplace= True) # I have set these frames to a high value as this enables all 'normal' events to not be registered within the anomaly ranges

df['Anomaly'] = np.where((df['starting frame of anomaly'] < df['frame_number']) & (df['ending frame of anomaly'] > df['frame_number']),1, 0)
df['Anomaly_Type'] = np.where((df['starting frame of anomaly'] < df['frame_number']) & (df['ending frame of anomaly'] > df['frame_number']),df['name'].str.split('_').str[0], "Normal")
df['scenario'] = df['scenario'].fillna(df['name'].apply(lambda x: re.sub(r'\d+', '', x))) # Due to alot of normal data not containing scenarios, these can be extracted from the name
df['scenario'] = df['scenario'].replace({"MSAD_normal_testing_":np.nan, "perdestrian_street":"pedestrian_street"}) # Removing MSAD normal testing at not relevant, corrected spelling mistakes.
df.to_csv("joined_table.csv")

print(df.info(),df.describe())
# Exploration of scenario and anomaly, where 1 represents anomaly occuring.
sns.countplot(df,x = "scenario", hue = 'Anomaly')
plt.title("Frames by Situation")
plt.xlabel('Scenario',fontdict={'size':10,'weight':'bold'})
plt.xticks(rotation = 'vertical')
plt.ylabel('Frames',fontdict={'size':10,'weight':'bold'})
plt.grid()
plt.show()

