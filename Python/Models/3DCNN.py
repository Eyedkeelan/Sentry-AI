import os
import pandas as pd 
import numpy as np 
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

BASE_DIR = r"C:\Users\Keelan.Butler\Desktop\python_projects\Final Project"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


ROUTE_DIR = r"C:\Users\Keelan.Butler\Desktop\python_projects\Final Project"
DATA_DIR = os.path.join(ROUTE_DIR,"Sentry-AI","Python")
FEATURE_PATH = os.path.join(BASE_DIR,"Dataset","features")

os.chdir(DATA_DIR)
df_train, df_test = pd.read_csv('Train_set.csv'),pd.read_csv('Test_set.csv')


# load features 
def load_features(dataframe):
    """Function extracts features from dataframe and processes them for modelling.
    returns: features, anomaly, anomaly_bool, frame_number
    Arg:
    - dataframe: This will be our test/train dataset. 
    Returns: 
    - Features: Loaded features from corresponding npy file. 
    - anon_type: A break down on what anomaly has occured, otherwise Normal. 
    - anon_bool: A boolean of anomaly occurance.
    - start_frame: List of the corresponding final frame of the sliding window
    - end_frame: List of the corresponding final frame of the sliding window
    - video_names: List of video names, this enables up to partition by videos.
    """
    features = []
    anon_type = []
    anon_bool = []
    start_index = []
    end_index = []
    video_names = []
    for path,subdir, label, lab_bool, clip_start, clip_end, video_name in tqdm(zip(dataframe['feature_path'],dataframe['subfolder'], dataframe['Anomaly_Type'], dataframe['Anomaly'], dataframe['Start_of_Clip'], dataframe['End_of_Clip'], dataframe['name'])):
        feature = np.load(os.path.join(FEATURE_PATH,subdir,path)) # Accesses the relevant path through our relational database 
        features.append(feature) # List of loaded npy features.
        anon_type.append(label) # List of type of anomalies. 
        anon_bool.append(lab_bool) # 1 for anomaly, 0 for normal.
        video_names.append(video_name) # List of video names.
        start_index.append(clip_start) # List of clip start frame
        end_index.append(clip_end) # List of clip start frame
    return features, anon_type, anon_bool, start_index,end_index, video_names

features, *_ = load_features(df_train)
load_features(df_test)


features[0].shape()


