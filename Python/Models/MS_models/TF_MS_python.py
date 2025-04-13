# %%
# END2END RESNET 
import os
import pandas as pd 
import numpy as np 
from tqdm import tqdm
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from transformers import VideoMAEForVideoClassification
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
import cv2
from PIL import Image


# %%
FRAME_INTERVAL = 5  # Capture every 5th frame
CLIP_LENGTH = 16  # Number of frames per clip for 3D CNN
FRAME_HEIGHT, FRAME_WIDTH = 224, 224  # r3d_18 with input 112x112 Slowfast 224x224


torch.cuda.is_available()

# %%
os.listdir()

# %%
project_path = r"C:\Users\Keelan24\Documents\My Projects\MSC Final" #Path to project destination

MSAD_File_Name = "OneDrive_2025-03-06"

VIDEO_DIR = os.path.join(project_path,MSAD_File_Name,"MSAD Dataset","MSAD_blur")
SAVE_DIR = os.path.join(project_path,"Processed_Frames")
Anomaly_dir = os.path.join(project_path,MSAD_File_Name, "MSAD Dataset","anomaly_annotation.csv")



# %%
transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((FRAME_HEIGHT, FRAME_WIDTH)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalising the features
    ])

# %%
Anomaly_data = pd.read_csv(Anomaly_dir)
anomalies = set([anon.split("_")[0] for anon in Anomaly_data.name.values])
print(f'Anomalies: {anomalies}')

anno_names = Anomaly_data.name.values.tolist()
anno_start = Anomaly_data['starting frame of anomaly'].values.tolist()
anno_end = Anomaly_data['ending frame of anomaly'].values.tolist()

# %%
# Save directory for extracted frames

def extract_and_save_frames(video_path, save_dir, frame_interval=5):
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_folder = os.path.join(save_dir, video_name)
    os.makedirs(save_folder, exist_ok=True)

    frame_count = 0
    saved_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        if frame_count % frame_interval == 0:
            # Convert BGR (OpenCV) to RGB (PIL)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Apply transformations
            frame = transform(frame)  # Now it's a Tensor (C, H, W)

            # Convert back to PIL image to save
            frame = transforms.ToPILImage()(frame)

            # Save frame as JPEG
            frame_path = os.path.join(save_folder, f"frame_{saved_count:04d}.jpg")
            frame.save(frame_path, "JPEG")
            saved_count += 1 

        frame_count += 1
    
    cap.release()

# %%
# # My iteration through
# for root, _, files in os.walk(VIDEO_DIR):
#     for video_file in tqdm(files, desc=f"Extracting Frames {root}"):
#         if video_file.endswith((".mp4", ".avi", ".mov")):
#             video_path = os.path.join(root, video_file)
#             extract_and_save_frames(video_path, SAVE_DIR)

# %%
anomaly = []
anonamly_bool = []
frame_paths = []
frames = []
video_names = []
video_path = []
import os
for root, _, files in os.walk(SAVE_DIR):
   for name in files:
      frame_path = os.path.join(root, name)
      components = frame_path.split(os.sep) 
      video_name =  components[-2]
      frame = int(components[-1].split("_")[1].split(".")[0]) * FRAME_INTERVAL
      frames.append(frame)
      video_names.append(video_name)
      #print(video_name) 
      frame_paths.append(frame_path)
      anom = video_name.split("_")[0]
      if anom in anomalies:
          #print(frame,video_name)
          pos = anno_names.index(video_name)
          start = anno_start[pos]
          end = anno_end[pos]

          if start < frame < end: 
              anon_bool = 1 
              anomaly_label = anom
          else:
              anon_bool = 0
              anomaly_label = "Normal"
         
      else:
          anon_bool = 0
          anomaly_label = "Normal"
      anomaly.append(anomaly_label)
      anonamly_bool.append(anon_bool)


metadata = pd.DataFrame({'Video':video_names,
              'Frame':frames,
             'Frames_path':frame_paths, 
             "Anomaly Type": anomaly,
             "Anomaly": anonamly_bool})

metadata["Video"] = metadata["Video"].str.replace("MSAD_normal_", "", regex=False)

# %%
with open(os.path.join(os.path.join(*os.path.split(VIDEO_DIR)[:-1]),"MSAD_I3D_WS_Train.list")) as train:
    t = train.readlines()
    train_list = [item.split("\n")[0].split("/")[-1].replace("_i3d.npy","") for item in t]
    train_label = ["Train"] * len(train_list)
tr_labels = pd.DataFrame({"Video":train_list,
                        "partition":train_label}) 
with open(os.path.join(os.path.join(*os.path.split(VIDEO_DIR)[:-1]),"MSAD_I3D_WS_Test.list")) as test:
    t = test.readlines()
    test_list = [item.split("\n")[0].split("/")[-1].replace("_i3d.npy","") for item in t]
    test_label = ["Test"] * len(test_list)

te_labels = pd.DataFrame({"Video":test_list,
                         "partition":test_label})
label_df = pd.concat([tr_labels,te_labels])
label_df["Video"] = label_df["Video"].str.replace("MSAD_normal_", "", regex=False)
label_df

# %%
df = pd.merge(left= metadata, right = label_df , on= "Video",how= "left")

l_encoder = LabelEncoder()
df["Anomaly Type"] = l_encoder.fit_transform(df["Anomaly Type"])

df_train =  df[df["partition"] == "Train"].drop(columns= "partition")
df_test =  df[df["partition"] == "Test"].drop(columns= "partition")

# %%
df_train

# %%
class CCTVFrameDataset(Dataset):
    def __init__(self, data, transform=None, sequence_length=16):
        self.data = data
        self.transform = transform
        self.sequence_length = sequence_length
        self.data.sort_values(by=['Video', 'Frame'], inplace=True)

    def __len__(self):
        return len(self.data) - self.sequence_length + 1  

    def __getitem__(self, idx):
        sequence = self.data.iloc[idx : idx + self.sequence_length]
    
        frames = []
        for _, row in sequence.iterrows():
            img_path = row['Frames_path']
            frame = Image.open(img_path).convert("RGB")  
            frame = np.array(frame, dtype=np.float32)  
            frame = torch.tensor(frame).permute(2, 0, 1) 

            if self.transform:
                frame = self.transform(frame)

            frames.append(frame)

        frames_tensor = torch.stack(frames)  # Shape: (T, C, H, W)
        label = sequence['Anomaly Type'].max()  # Assign anomaly type in sequence
    
        return frames_tensor, torch.tensor(label, dtype=torch.long)


# %%
from collections import Counter

# Build a temporary dataset to extract labels
temp_dataset = CCTVFrameDataset(df_train, sequence_length=16)

# Extract all labels from the training dataset
sequence_labels = [temp_dataset[i][1].item() for i in range(len(temp_dataset))]

# Count how often each label appears
class_counts = Counter(sequence_labels)
num_classes = len(class_counts)

# Inverse of frequency as weight
class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}

# Assign weights to each sample
sample_weights = [class_weights[label] for label in sequence_labels]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights), 
    replacement=True)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = os.cpu_count() - 1  # Set number of workers

batch_size = 32 if device.type == 'cuda' else 8  # Larger batch size for GPU

training_dataset = CCTVFrameDataset(df_train)
testing_dataset = CCTVFrameDataset(df_test)
print(f"Training dataset size: {len(training_dataset)}\nTesting dataset size: {len(testing_dataset)}")

training_dataloader = torch.utils.data.DataLoader(
    training_dataset,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=0,
    pin_memory=(device.type == 'cuda')
)

testing_dataloader = torch.utils.data.DataLoader(
    testing_dataset, batch_size=batch_size, shuffle=True,
    num_workers=0, pin_memory=(device.type == 'cuda')
)

print(f"Dataloader initialized with {'pin_memory' if device.type == 'cuda' else 'no pin_memory'} and {num_workers} workers")

# %%
import time

start_time = time.time()
batch = next(iter(training_dataloader))  # Try loading a single batch
end_time = time.time()

print(f"Time to load one batch: {end_time - start_time:.2f} seconds")


# %%
print(training_dataset[0][0].shape)  

# %%
# model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400") # Facebooks video classification model - proved too large for efficient training
# model.classifier = nn.Linear(in_features=768, out_features=1, bias=True)

model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base")
model.classifier = nn.Linear(in_features=768, out_features=12, bias=True)


# %%
print(model)

# %%
from torch.cuda.amp import autocast

# Training Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ms = model

epochs = 5
losses = np.zeros((2, epochs))
model_ms.to(device)

optimiser = optim.Adam(model_ms.parameters(), lr=1e-6, weight_decay=1e-4)  
loss_function = nn.BCEWithLogitsLoss()

# Use GPU if available

print(f"Using device: {device}")


best_loss = np.inf

threshhold = 0.5 

# Training Loop
for epoch in range(epochs):
    epoch_loss = 0.0
    model_ms.train()

    for frames, labels in tqdm(training_dataloader, desc=f"Training epoch {epoch+1}/{epochs}"):
        torch.cuda.empty_cache()  

        frames, labels = frames.to(device), labels.to(device)

        # TimeSformer expects input: (batch_size, num_frames, channels, height, width)
       # frames = frames.permute(0, 1, 4, 2, 3)  # Swap channel & frame dims if necessary

        # Forward pass
        with autocast():
            outputs = model_ms(frames).logits  # Extract logits

        # Compute loss
        loss = loss_function(outputs.squeeze(), labels.float())

        # Backpropagation
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        epoch_loss += loss.item()
    
    losses[0, epoch] = epoch_loss / len(training_dataloader)

    # Validation Loop
    model_ms.eval()
    test_loss = 0.0

    with torch.no_grad():
        for test_frames, test_labels in tqdm(testing_dataloader, desc="Validating"):
            test_frames, test_labels = test_frames.to(device), test_labels.to(device)
            #test_frames = test_frames.permute(0, 2, 1, 3, 4)  

            test_preds = model_ms(test_frames).logits  
            t_loss = loss_function(test_preds.squeeze(), test_labels.float())

            test_loss += t_loss.item()

    losses[1, epoch] = test_loss / len(testing_dataloader)

    # Save best model
    if best_loss > losses[1, epoch]:
        best_loss = losses[1, epoch]
        print(f"Saving best model at epoch {epoch + 1}")
        torch.save(model_ms.state_dict(), os.path.join("Best_Models", "TimeSformer_ms.pt"))

    print(f"Epoch [{epoch+1}/{epochs}] - Training Loss: {losses[0,epoch]:.4f}, Test Loss: {losses[1,epoch]:.4f}")

# %%
plt.plot(losses[0], label = 'Training')
plt.plot(losses[1], label = 'Testing')
plt.grid()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("TimeSformer Multi-scenario Classification Model")
plt.savefig("")

# %%
model_ms.load_state_dict(torch.load(os.path.join("Best_Models", "TimeSformer_ms.pt")))
print("Best model loaded!")

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_predictions(model, dataloader, device):
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():  # Disable gradient computation for efficiency
        for frames, labels in tqdm(dataloader):
            frames = frames
            labels = labels
            slow_frames, fast_frames = frames  # Unpack SlowFast inputs
        
            slow_frames, fast_frames, labels = (
                slow_frames.to(device),
                fast_frames.to(device),
                labels.to(device),
            )

            slow_frames = slow_frames.permute(0, 4, 2, 3, 1)
            fast_frames = fast_frames.permute(0, 4, 2, 3, 1)
            outputs = model(slow_frames, fast_frames)  # Forward pass
           # _, preds = torch.max(outputs, 1)  # Get predicted class MULTICLASS APPROACH
            #preds = (outputs >= 0.5).float() # Binalry class 
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            all_preds.extend(preds.cpu().numpy())  # Store predictions
            all_labels.extend(labels.cpu().numpy())  # Store true labels

    return np.array(all_labels), np.array(all_preds)

# Example usage:
device = torch.device(device)  # Change to "cuda" if using GPU
true_labels, pred_labels = get_predictions(model_ms, testing_dataloader, device)
print(f"Total test samples: {len(true_labels)} (Expected: 29270)")
print(f"Total predictions: {len(pred_labels)} (Expected: 29270)")


# %%
cm = metrics.confusion_matrix(true_labels, pred_labels)
sns.heatmap(cm , annot = True)
plt.title("Confusion Matrix of Multi-scenario Transformer Classifier")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig('ms_tf_cm.png')
plt.show()

# %%



