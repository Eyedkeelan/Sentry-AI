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
project_path = r"/mnt/scratch/od22kob/MSC Final"

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
# My iteration through
for root, _, files in os.walk(VIDEO_DIR):
    for video_file in tqdm(files, desc=f"Extracting Frames {root}"):
        if video_file.endswith((".mp4", ".avi", ".mov")):
            video_path = os.path.join(root, video_file)
            extract_and_save_frames(video_path, SAVE_DIR)

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
df_train =  df[df["partition"] == "Train"].drop(columns= "partition")
df_test =  df[df["partition"] == "Test"].drop(columns= "partition")

# %%
class FrameDataset(Dataset):
    def __init__(self, labels_df, transform=None, clip_length=32, tau=4, step_size=8):
        """
        labels_df: DataFrame with columns ["Video", "Frame", "Frames_path", "Anomaly Type", "Anomaly"]
        transform: Image transformations (for resizing, normalizing, etc.)
        clip_length: Number of frames per sample for Fast Path (default 32)
        tau: Frame stride for Slow Path (default 4, meaning every 4th frame)
        step_size: How far the window moves per sample (default 8)
        """
        self.labels_df = labels_df.copy()  # Prevent modifying the original DataFrame
        self.transform = transform
        self.clip_length = clip_length
        self.tau = tau
        self.step_size = step_size

        # Ensure data is sorted by video and frame number
        self.labels_df.sort_values(by=["Video", "Frame"], inplace=True)

        # Group frames by video
        self.video_groups = self.labels_df.groupby("Video")

        # Encode labels
        self.label_encoder = LabelEncoder()
        self.labels_df["Encoded_Label"] = self.label_encoder.fit_transform(self.labels_df["Anomaly"])

        # Store unique videos
        self.video_list = list(self.video_groups.groups.keys())

    def __len__(self):
        total_clips = 0
        for video_name in self.video_list:
            num_frames = len(self.video_groups.get_group(video_name))
            num_clips = max(0, (num_frames - self.clip_length) // self.step_size + 1)
            total_clips += num_clips
        return total_clips

    def __getitem__(self, idx):
        # Determine video & clip index
        total_clips = 0
        for video_name in self.video_list:
            video_frames = self.video_groups.get_group(video_name)
            num_frames = len(video_frames)
            num_clips = max(0, (num_frames - self.clip_length) // self.step_size + 1)

            if idx < total_clips + num_clips:
                clip_idx = idx - total_clips
                break

            total_clips += num_clips
        else:
            raise IndexError(f"Index {idx} is out of range for dataset of length {self.__len__()}")

        # Get frames & labels
        frame_paths = video_frames["Frames_path"].tolist()
        frame_labels = video_frames["Encoded_Label"].tolist()

        start_idx = clip_idx * self.step_size

        # Select frames for **Fast Path**
        fast_frames_paths = frame_paths[start_idx:start_idx + self.clip_length]
        fast_labels = frame_labels[start_idx:start_idx + self.clip_length]

        # Select frames for **Slow Path** (every `tau` frames from fast frames)
        slow_frames_paths = fast_frames_paths[::self.tau]
        slow_labels = fast_labels[::self.tau]

        # Ensure Slow Path has enough frames
        while len(slow_frames_paths) < self.clip_length // self.tau:
            slow_frames_paths.append(slow_frames_paths[-1])
            slow_labels.append(slow_labels[-1])

        # Load images
        def load_frames(paths):
            frames = []
            for frame_path in paths:
                frame = cv2.imread(frame_path)
                if frame is None:
                    raise ValueError(f"Error reading image: {frame_path}")

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if self.transform:
                    frame = self.transform(frame)
                else:
                    frame = torch.from_numpy(frame).float()

                frames.append(frame)

            return torch.stack(frames)  # Shape: (T, C, H, W)

        fast_frames = load_frames(fast_frames_paths)
        slow_frames = load_frames(slow_frames_paths)

        # Transpose to match SlowFast input format (C, T, H, W)
        fast_frames = fast_frames.permute(1, 0, 2, 3)  # (C, T, H, W)
        slow_frames = slow_frames.permute(1, 0, 2, 3)  # (C, T/4, H, W)

        # Assign majority label
        clip_label = torch.tensor(fast_labels).mode()[0]  # Get most frequent label

        return [slow_frames, fast_frames], torch.tensor(clip_label, dtype=torch.long)


# %%
training_dataset = FrameDataset(labels_df = df_train)
testing_dataset = FrameDataset(labels_df = df_test)
print("Training dataset size: {}\ntesting dataset size: {}".format(len(training_dataset),len(testing_dataset)))
num_workers = os.cpu_count() - 1


clip_labels = []
for i in range(len(training_dataset)):
    _, label = training_dataset[i]
    clip_labels.append(label.item())



from collections import Counter

label_counts = Counter(clip_labels)
total = sum(label_counts.values())
class_weights = {cls: total / count for cls, count in label_counts.items()}

weights = [class_weights[label] for label in clip_labels]

sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)



training_dataloader = torch.utils.data.DataLoader(
    training_dataset,
    batch_size=8,
    sampler=sampler,
    num_workers=num_workers
)
testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size=8, shuffle=True, num_workers=num_workers)#, pin_memory=True)






# %%
slowfast_model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)

# Remove classification head
slowfast_model.blocks[-1] = nn.Identity() # Remove the final classification layer

class SlowFastBinaryClassifier(nn.Module):
    def __init__(self, slowfast_model):
        super(SlowFastBinaryClassifier, self).__init__()
        self.slowfast_model = slowfast_model
        self.fc = nn.Linear(2304 , 1)  # Output 1 for binary classification

    def forward(self, slow_frames, fast_frames):
        features = self.slowfast_model([slow_frames, fast_frames])  # Forward pass
        #print("Feature shape before pooling:", features.shape)
        
        features = F.adaptive_avg_pool3d(features, (1, 1, 1)).squeeze()

        return self.fc(features)

# %%
model_binary = SlowFastBinaryClassifier(slowfast_model)

# %%
# Training Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_binary = SlowFastBinaryClassifier(slowfast_model)

epochs = 50
losses = np.zeros((2, epochs))
model_binary.to(device)

optimiser = optim.Adam(model_binary.parameters(), lr=1e-7, weight_decay=1e-4)  
loss_function = nn.BCEWithLogitsLoss()

# Use GPU if available

print(f"Using device: {device}")


best_loss = np.inf

threshhold = 0.5 

# Training Loop
for epoch in range(epochs):
    epoch_loss = 0.0
    model_binary.train()

    for frames, labels in tqdm(training_dataloader, desc=f"Training pass epoch: {epoch}"):
        torch.cuda.empty_cache()  
        slow_frames, fast_frames = frames  # Unpack SlowFast inputs
        
        slow_frames, fast_frames, labels = (
            slow_frames.to(device),
            fast_frames.to(device),
            labels.to(device),
        )

        slow_frames = slow_frames.permute(0, 4, 2, 3, 1)
        fast_frames = fast_frames.permute(0, 4, 2, 3, 1)
        
        
        #print("slow_frames shape:", slow_frames.shape)
        #print("fast_frames shape:", fast_frames.shape)

        pred = model_binary(slow_frames, fast_frames)
        loss = loss_function(pred.squeeze(), labels.float())

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
        epoch_loss += loss.item()
        torch.cuda.empty_cache()  
    # Store training loss
    losses[0, epoch] = epoch_loss / len(training_dataloader)

    # Validation Loop
    model_binary.eval()
    test_loss = 0.0

    with torch.no_grad():
        for test_frames, test_labels in tqdm(testing_dataloader, desc="Cycling Testing Dataloader"):
            slow_test_frames, fast_test_frames = test_frames  # Unpack test data
            
            slow_test_frames, fast_test_frames, test_labels = (
                slow_test_frames.to(device),
                fast_test_frames.to(device),
                test_labels.to(device),
            )
            slow_test_frames = slow_test_frames.permute(0, 4, 2, 3, 1)
            fast_test_frames = fast_test_frames.permute(0, 4, 2, 3, 1)

            test_preds = model_binary(slow_test_frames, fast_test_frames)  
            t_loss = loss_function(test_preds.squeeze(), test_labels.float())

            test_loss += t_loss.item()
            
    losses[1, epoch] = test_loss / len(testing_dataloader)

    # Save best model
    if best_loss > losses[1, epoch]:
        best_loss = losses[1, epoch] 
        print(f"Saving Optimal model: {epoch + 1} epoch")
        torch.save(model_binary.state_dict(), os.path.join("Best_Models", "E2E_SF.pt"))

    print(f"Epoch [{epoch+1}/{epochs}] - Training Loss: {losses[0,epoch]:.4f}, Test Loss: {losses[1,epoch]:.4f}")

# %%
plt.plot(losses[0], label = 'Training')
plt.plot(losses[1], label = 'Testing')
plt.grid()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Slow-Fast Binary Classification Model")
plt.savefig("Training_E2E_SF_Binary.png")
plt.show()

# %%
model_binary.load_state_dict(torch.load(os.path.join("Best_Models", "E2E_SF.pt")))
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

device = torch.device(device)  # Change to "cuda" if using GPU
true_labels, pred_labels = get_predictions(model_binary, testing_dataloader, device)
print(f"Total test samples: {len(true_labels)} (Expected: 29270)")
print(f"Total predictions: {len(pred_labels)} (Expected: 29270)")


# %%
cm = metrics.confusion_matrix(true_labels, pred_labels)
sns.heatmap(cm , annot = True)
plt.title("Confusion Matrix of Slow-Fast Classifier")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("E2E_SF_Binary_results.png")
plt.show()

# %%



