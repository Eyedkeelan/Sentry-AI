import os
import pandas as pd 
import numpy as np 
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


ROUTE_DIR = r"C:\Users\Keelan.Butler\Desktop\python_projects\Final Project"
DATA_DIR = os.path.join(ROUTE_DIR,"Sentry-AI","Python")
os.chdir(DATA_DIR)
df_train, df_test = pd.read_csv('Train_set.csv'),pd.read_csv('Test_set.csv')


# load features 
def load_features(dataframe):
    """Function extracts features from dataframe and processes them for frame sequencing.
    returns: features, anomaly, anomaly_bool, frame_number
    Arg:
    - dataframe: This will be our test/train dataset. 
    Returns: 
    - Features: Loaded features from corresponding npy file. 
    - anon_type: A break down on what anomaly has occured, otherwise Normal. 
    - anon_bool: A boolean of anomaly occurance.
    - frame_num: List of the corresponding frames. 
    - video_names: List of video names, this enables up to partition by videos.
    """
    features = []
    anon_type = []
    anon_bool = []
    frame_num = []
    video_names = []
    for path, label, lab_bool, frame_pos, video_name in tqdm(zip(dataframe['feature_path'], dataframe['Anomaly_Type'], dataframe['Anomaly'], dataframe['frame_number'], dataframe['name'])):
        feature = np.load(path) # Accesses the relevant path through our relational database 
        features.append(feature) # List of loaded npy features.
        anon_type.append(label) # List of type of anomalies. 
        anon_bool.append(lab_bool) # 1 for anomaly, 0 for normal.
        video_names.append(video_name) # List of video names.
        frame_num.append(frame_pos) # Frame number relevant to the video. 
    return features, anon_type, anon_bool, frame_num, video_names

def sequence_frames(features, anon_type, anon_bool, frame_no, video_names, window_size=30, step=15):
    """ 
    Convert individual frame features into sequences.
    
    Args:
    - features: List of numpy arrays representing frame-level features.
    - anon_type: List of anomaly types.
    - anon_bool: List of binary anomaly labels.
    - frame_no: List of frame numbers.
    - video_names: List of video names.
    - window_size: Number of frames per sequence.
    - step: Step size for the sliding window.
    
    Returns:
    - seq_features: List of feature sequences of shape (window_size, C).
    - seq_anon_type: Corresponding anomaly types.
    - seq_anon_bool: Anomaly labels (1 if at least one frame in sequence is anomalous).
    - seq_video_names: Video names for each sequence.
    """
    seq_features = []
    seq_anon_type = []
    seq_anon_bool = []
    seq_video_names = []

    video_df = pd.DataFrame({'features': features, 'anon_type': anon_type, 'anon_bool': anon_bool, 'frame_no': frame_no, 'video_name': video_names})
    
    # Group by video name to ensure frames are processed in correct order
    for video, group in tqdm(video_df.groupby('video_name'), desc = "Grouping by video and sequencing. "):
        group = group.sort_values(by='frame_no')  # Sort by frame number
        video_features = np.stack(group['features'].values)  # Stack features into (num_frames, C)
        video_anon_type = group['anon_type'].values
        video_anon_bool = group['anon_bool'].values
        
        # Create sequences using a sliding window approach
        for i in range(0, len(video_features) - window_size + 1, step):
            seq_features.append(video_features[i:i+window_size])  # Shape: (window_size, C)
            seq_anon_type.append(video_anon_type[i])  # Assign anomaly type based on the first frame
            seq_anon_bool.append(1 if np.any(video_anon_bool[i:i+window_size]) else 0)  # Mark as anomaly if any frame is anomalous
            seq_video_names.append(video)
    
    return seq_features, seq_anon_type, seq_anon_bool, seq_video_names

train_features, train_anon_type, train_anon_bool, train_frame_no, train_video_name = load_features(df_train)

test_features, test_anon_type, test_anon_bool, test_frame_no, test_video_name = load_features(df_test)
print(f"features: {np.array(train_features).shape} featureshape: {np.array(train_features[0]).shape}")
# Process training sequences
train_seq_features, train_seq_anon_type, train_seq_anon_bool, train_seq_video_names = sequence_frames(
    train_features, train_anon_type, train_anon_bool, train_frame_no, train_video_name)

# Process testing sequences
test_seq_features, test_seq_anon_type, test_seq_anon_bool, test_seq_video_names = sequence_frames(
    test_features, test_anon_type, test_anon_bool, test_frame_no, test_video_name)

class VideoDataset(Dataset):
    def __init__(self, features, labels):
        """Args: 
        - features: List of arrays with shape (T, c)
        - labels: List of anomaly labels(0: Normal, 1: Anomaly)"""
        self.features = [torch.tensor(f, dtype=torch.float32).permute(1, 0).unsqueeze(-1).unsqueeze(-1) for f in features] 
        # Shape: (C, T, 1, 1)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self,):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
print(np.array(train_seq_features).shape)
train_dataset = VideoDataset(train_seq_features, train_seq_anon_bool)
test_dataset = VideoDataset(test_seq_features, test_seq_anon_bool)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class VideoAnomalyDetector(nn.Module):
    def __init__(self, in_channels):
        super(VideoAnomalyDetector, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=(3, 3, 3), padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d((2, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d((2, 2, 2))

        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=1)
        self.bn3 = nn.BatchNorm3d(256)
        self.pool3 = nn.AdaptiveAvgPool3d((1, 1, 1))  # Global pooling

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)  # Binary classification

        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))

        x = x.view(x.shape[0], -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x



model = VideoAnomalyDetector(in_channels=train_seq_features[0].shape[1]).to(device)  # Use the feature dimension as input channels

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device).unsqueeze(1)  # Reshape labels for BCE Loss
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}")
