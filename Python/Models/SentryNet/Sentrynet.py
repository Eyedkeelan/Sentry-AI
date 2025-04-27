#!/usr/bin/env python
# coding: utf-8

# In[17]:


import json
import os
import pandas as pd 
import numpy as np 
from tqdm import tqdm
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
import cv2
from PIL import Image
import re
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import ast



Training = True

# In[2]:


# Extract and process captions

ROOT = r"/mnt/scratch/od22kob/"
JSON_PATHS = os.path.join(ROOT,"Capdata","All the annotation","All the annotation")
IMAGE_FOLDERS =  os.path.join(ROOT,"Capdata","UCF crime Extrated frame Dataset","UCF crime Extrated frame Dataset")
os.listdir(JSON_PATHS)


# In[3]:


def extract_json(json_path, JSON_PATHS = JSON_PATHS):
    p = os.path.join(JSON_PATHS, json_path)
    with open(p, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame.from_dict(data, orient='index')
    df.index.name = 'video_id'

    return df

df = pd.DataFrame()
for i, json_file in enumerate(os.listdir(JSON_PATHS)):
    df = pd.concat([df,extract_json(json_file)])

df.reset_index(inplace=True)


# In[4]:


expanded_rows = []

for _, row in df.iterrows():
    video_id = row["video_id"]
    for (start, end), sentence in zip(row["timestamps"], row["sentences"]):
        expanded_rows.append({
            "video_id": video_id,
            "start_time": start,
            "end_time": end,
            "caption": sentence
        })
expanded_df = pd.DataFrame(expanded_rows)

captions_df = expanded_df

captions_df


# In[5]:


# Tokenize all captions into words
all_tokens = [word.lower() for cap in expanded_df["caption"] for word in cap.split()]

# Add special tokens
specials = ["<pad>", "<unk>", "<sos>", "<eos>"]
word_counts = Counter(all_tokens)
vocab = {token: idx for idx, token in enumerate(specials + list(word_counts.keys()))}
inv_vocab = {idx: token for token, idx in vocab.items()}

inv_vocab.get(9)


# In[6]:


# Frames

EFFECTIVE_FPS = 3
CLIP_LEN = 32
FRAME_HEIGHT, FRAME_WIDTH = 224, 224
MAX_LENGTH = 50
transform = T.Compose([
        T.Resize((FRAME_HEIGHT, FRAME_WIDTH)), 
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Same transform as used in classification
    ])



# In[7]:


records = []

for class_name in os.listdir(IMAGE_FOLDERS):
    class_path = os.path.join(IMAGE_FOLDERS, class_name)
    if not os.path.isdir(class_path):
        continue

    for file_name in os.listdir(class_path):
        if not file_name.endswith(".png"):
            continue

        # Example: Fighting002_x264_frame_390.png
        parts = file_name.split("_frame_")
        video_id = parts[0]
        frame_idx = int(parts[1].split(".")[0])

        # Build a mapping of video_id -> all frame indices
        # We'll group frames belonging to the same video
        records.append({
            "class_name": class_name,
            "video_id": video_id,
            "frame_idx": frame_idx
        })

# Turn into DataFrame
frame_df = pd.DataFrame(records)


# In[8]:


clip_records = []

for (class_name, video_id), group in tqdm(frame_df.groupby(["class_name", "video_id"])):
    frame_indices = sorted(group["frame_idx"].tolist())

    # Slide window over frames
    for i in range(0, len(frame_indices) - CLIP_LEN + 1, CLIP_LEN):
        clip_frames = frame_indices[i:i+CLIP_LEN]

        start_frame = clip_frames[0]
        end_frame = clip_frames[-1]

        start_time = start_frame / EFFECTIVE_FPS
        end_time = end_frame / EFFECTIVE_FPS

        clip_records.append({
            "class_name": class_name,
            "video_id": video_id,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time": start_time,
            "end_time": end_time,
            "clip_frames": clip_frames  # (optional: list of all frames)
        })
clip_df = pd.DataFrame(clip_records)
clip_df 


# In[9]:


matched_clips = []

for idx, clip_row in tqdm(clip_df.iterrows()):
    clip_vid = clip_row['video_id']
    clip_start = clip_row['start_time']
    clip_end = clip_row['end_time']

    # Find captions for the same video where there is ANY overlap
    mask = (captions_df['video_id'] == clip_vid) & \
           (captions_df['start_time'] < clip_end) & \
           (captions_df['end_time'] > clip_start)

    matching_captions = captions_df[mask]['caption'].tolist()

    matched_clips.append({
        "class_name": clip_row['class_name'],
        "video_id": clip_vid,
        "start_frame": clip_row['start_frame'],
        "end_frame": clip_row['end_frame'],
        "start_time": clip_start,
        "end_time": clip_end,
        "clip_frames": clip_row['clip_frames'],
        "captions": matching_captions  # Overlapping captions
    })


# In[10]:


df = pd.DataFrame(matched_clips)

df = df[df['captions'].map(lambda x: len(x) > 0)]
df = df.reset_index(drop=True)
df = df.explode('captions').reset_index(drop=True)
df


# In[11]:


nltk.download('punkt_tab')
def tokenize(text):
    return word_tokenize(text.lower())

all_tokens = [token for cap in expanded_df["caption"] for token in tokenize(cap)]

specials = ["<pad>", "<unk>", "<sos>", "<eos>"]
word_counts = Counter(all_tokens)

filtered_tokens = [word for word, freq in word_counts.items() if freq > 3]

vocab = {token: idx for idx, token in enumerate(specials + filtered_tokens)}
inv_vocab = {idx: token for token, idx in vocab.items()}


# In[12]:


def simple_tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

# Tokenize every caption (each row now has just one caption)
df['tokens'] = df['captions'].apply(lambda x: simple_tokenize(x))

def tokens_to_ids(tokens, vocab):
    sos = vocab["<sos>"]
    eos = vocab["<eos>"]
    unk = vocab["<unk>"]

    ids = [sos] + [vocab.get(token, unk) for token in tokens] + [eos]
    return ids

# Convert tokens to token_ids
df['token_ids'] = df['tokens'].apply(lambda tokens: tokens_to_ids(tokens, vocab))

df


# In[13]:


class VideoCaptionDataset(Dataset):
    def __init__(self, caption_data, video_dir, max_seq_len=64, pad_id=0):
        self.data = caption_data
        self.video_dir = video_dir
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id
        
        # Fix data types (because clip_frames, tokens, token_ids are saved as strings)
        self.data['clip_frames'] = self.data['clip_frames']#.apply(ast.literal_eval)
        self.data['token_ids'] = self.data['token_ids']#.apply(ast.literal_eval)

        # Image transform for frame preprocessing
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet mean/std
        ])

    def __len__(self):
        return len(self.data)

    def pad_sequence(self, ids):
        if len(ids) >= self.max_seq_len:
            return ids[:self.max_seq_len]
        else:
            return ids + [self.pad_id] * (self.max_seq_len - len(ids))

    def load_frames(self, video_id, frame_indices):
        frames = []
        for idx in frame_indices:
            folder = re.match(r"([A-Za-z]+)\d+_x264", video_id).group(1)
            frame_path = os.path.join(self.video_dir,folder , f"{video_id}_frame_{idx}.png")  # Assuming images are stored as jpg files
            frame = Image.open(frame_path).convert('RGB')
            frame = self.transform(frame)
            frames.append(frame)
        return torch.stack(frames)

    def extract_slowfast_features(self, frames, slow_sample_rate=4):
        # Fast pathway: use all frames
        fast_pathway = frames
        
        # Slow pathway: sample frames with slow_sample_rate
        slow_pathway = frames[::slow_sample_rate]  # Take every 'slow_sample_rate' frame
    
        
        return slow_pathway, fast_pathway

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # === Load SlowFast features ===
        video_id = row['video_id']
        frame_indices = row['clip_frames']
        
        frames = self.load_frames(video_id, frame_indices)
        
        # Assuming SlowFast model gives us slow and fast pathways
        slow_pathway, fast_pathway = self.extract_slowfast_features(frames)

        # === Caption ===
        token_ids = self.pad_sequence(row['token_ids'])
        token_ids = torch.tensor(token_ids, dtype=torch.long)

        return (slow_pathway, fast_pathway), token_ids


# In[14]:


dataset = VideoCaptionDataset(
    df, 
    IMAGE_FOLDERS, 
    max_seq_len=64)

video_features, token_ids = dataset[420]


train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)

# Create subsets for training and testing
train_subset = Subset(dataset, train_indices)
test_subset = Subset(dataset, test_indices)
num_workers = 12
# Create DataLoaders
batch_size = 16

train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Check one batch from the train_loader
for video_features, token_ids in train_loader:
    print("Video Features (Slow Pathway) Shape:", video_features[0].shape)
    print("Video Features (Fast Pathway) Shape:", video_features[1].shape)
    print("Token IDs Shape:", token_ids.shape)
    break  # Just print the first batch and stop

# Optional: Verify the length of the dataset and the dataloaders
print(f"Length of full dataset: {len(dataset)}")
print(f"Length of train loader: {len(train_loader)}")
print(f"Length of test loader: {len(test_loader)}")






# An example to test outputs work
print(video_features[0].shape)  # slow pathway
print(video_features[1].shape)  # fast pathway
print(token_ids)
print(len(dataset))


# In[15]:

class Attention(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(feature_dim + hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, features, hidden):
        """
        features: (batch_size, seq_len, feature_dim)
        hidden: (batch_size, hidden_dim)
        """
        batch_size = features.size(0)
        seq_len = features.size(1)

        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, seq_len, hidden_dim)
        energy = torch.tanh(self.attn(torch.cat((features, hidden), dim=2)))  # (batch_size, seq_len, hidden_dim)
        attention = self.v(energy).squeeze(2)  # (batch_size, seq_len)
        attn_weights = torch.softmax(attention, dim=1)  # (batch_size, seq_len)

        context = torch.bmm(attn_weights.unsqueeze(1), features).squeeze(1)  # (batch_size, feature_dim)

        return context, attn_weights
        

class SentryNet(nn.Module): 
    def __init__(self, lstm_hidden_size, vocab_size, embedding_dim, num_layers=1, freeze_sf=True):
        super(SentryNet, self).__init__()
        # SlowFast encoder
        self.sf = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
        self.sf.blocks[-1] = nn.Identity()
        if freeze_sf: 
            for param in self.sf.parameters():
                param.requires_grad = False

        # Feature encoder LSTM
        self.encoder_lstm = nn.LSTM(input_size=2304, hidden_size=lstm_hidden_size, num_layers=num_layers, batch_first=True, dropout=0.3)

        # Attention module
        self.attention = Attention(feature_dim=2304, hidden_dim=lstm_hidden_size)

        # Caption decoder
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.decoder_lstm = nn.LSTM(input_size=embedding_dim + 2304, hidden_size=lstm_hidden_size, num_layers=num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(lstm_hidden_size, vocab_size)

    def forward(self, x, captions=None, teacher_forcing_ratio=0.5):
        batch_size = x[0].size(0)  # x is a tuple (slow, fast)
    
        # Extract slow and fast features
        slow_features = x[0].permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        fast_features = x[1].permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
    
        # Forward through SlowFast
        features = self.sf([slow_features, fast_features])  # (batch_size, 2304)
        features = features.flatten(1)  # (batch_size, 2304)
    
        features = features.unsqueeze(1)  # (batch_size, 1, 2304)
        
        # Ensure 'features' is properly initialized before being used
        if features is None:
            raise ValueError("The 'features' variable is None. Check the SlowFast model output.")
        
        # Encode features
        _, (h_n, c_n) = self.encoder_lstm(features)
    
        # Prepare decoder
        outputs = []
        device = x[0].device
        input_token = torch.full((batch_size,), vocab["<sos>"], dtype=torch.long, device=device)
    
        seq_len = captions.size(1) if captions is not None else 20
        features = features.repeat(1, seq_len, 1)  # Repeat for attention across sequence
    
        # new: keep track of which sequences have ended
        ended = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
        for t in range(seq_len):
            embedded = self.embedding(input_token).unsqueeze(1)  # (batch_size, 1, embedding_dim)
    
            # Attention context
            context, attn_weights = self.attention(features, h_n[-1])
    
            # Combine embedded token and attention context
            decoder_input = torch.cat([embedded.squeeze(1), context], dim=1).unsqueeze(1)  # (batch_size, 1, embedding_dim + 2304)
    
            # Pass through Decoder LSTM
            output, (h_n, c_n) = self.decoder_lstm(decoder_input, (h_n, c_n))  # (batch_size, 1, hidden_size)
            logits = self.fc(output.squeeze(1))  # (batch_size, vocab_size)
            outputs.append(logits)
    
            # next token prediction
            next_token = logits.argmax(1)
    
            # mark sequences that just generated <eos>
            ended = ended | (next_token == vocab["<eos>"])
    
            # once ended, feed <pad> thereafter
            input_token = next_token.clone()
            input_token[ended] = vocab["<pad>"]
    
            # teacher forcing override
            if captions is not None and torch.rand(1).item() < teacher_forcing_ratio:
                input_token = captions[:, t]
    
        return torch.stack(outputs, dim=1)  # (batch_size, seq_len, vocab_size)


lstm_hidden_size = 512
vocab_size = len(vocab)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

model = SentryNet(lstm_hidden_size=lstm_hidden_size, num_layers=3, vocab_size=vocab_size, embedding_dim=256).to(device)


# In[16]:


# Get first batch from train_loader
video_features, token_ids = next(iter(train_loader))
token_ids = token_ids.to(device)
video_features = (video_features[0].to(device), video_features[1].to(device))
print(video_features[1].shape)  # Should be [batch_size, channels, frames, height, width]
print(token_ids.shape)

# Pass through the models
output = model(video_features, token_ids)

smooth_fn = SmoothingFunction().method1
# In[18]:

if Training: 
  # Setup
  optimiser = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', patience=5, factor=0.1)
  
  criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token
  
  
  num_epochs = 100
  losses = np.zeros((2, num_epochs))
  bleu_scores = np.zeros(num_epochs)
  best_loss = float('inf')
  best_bleu = 0.0 
  
  for epoch in range(num_epochs):
      model.train()  
      total_loss = 0
  
      for video_features, token_ids in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
          video_features = [feature.to(device) for feature in video_features]
          token_ids = token_ids.to(device)
  
          optimiser.zero_grad()
  
          output = model(video_features, token_ids)
  
          # Reshape outputs
          output = output.view(-1, vocab_size)
  
          mask = token_ids != 0  # Exclude <pad> (0)
          mask = mask & (token_ids != 2)  # Exclude <sos> (2)
          output = output[mask.view(-1)]
          token_ids_flat = token_ids.view(-1)
          token_ids_flat = token_ids_flat[mask.view(-1)]
  
          loss = criterion(output, token_ids_flat)
          loss.backward()
          optimiser.step()
  
          total_loss += loss.item()
  
      avg_train_loss = total_loss / len(train_loader)
      losses[0, epoch] = avg_train_loss
  
      # Validation
      model.eval()
      val_loss = 0.0
      with torch.no_grad():
          for video_features, token_ids in tqdm(test_loader, desc=f"Validation Epoch {epoch+1}"):
              video_features = [feature.to(device) for feature in video_features]
              token_ids = token_ids.to(device)
  
              output = model(video_features, token_ids)
  
              output = output.view(-1, vocab_size)
  
              mask = token_ids != 0  # Exclude <pad> (0)
              mask = mask & (token_ids != 2)  # Exclude <sos> (2)
              output = output[mask.view(-1)]
              token_ids_flat = token_ids.view(-1)
              token_ids_flat = token_ids_flat[mask.view(-1)]
  
              loss = criterion(output, token_ids_flat)
              val_loss += loss.item()
  
      avg_val_loss = val_loss / len(test_loader)
      losses[1, epoch] = avg_val_loss
      scheduler.step(avg_val_loss)
      # Save best model
      if avg_val_loss < best_loss:
          best_loss = avg_val_loss
          print(f"Saving best model at epoch {epoch + 1}")
          torch.save(model.state_dict(), "/mnt/scratch/od22kob/MSC Final/Sentry-AI/Python/Models/Best_Models/SentryNet_best.pt")
  
      references = []
      hypotheses = []
      
      with torch.no_grad():
          for video_features, token_ids in tqdm(test_loader, desc=f"BLEU Epoch {epoch+1}"):
              video_features = [feature.to(device) for feature in video_features]
              token_ids = token_ids.to(device)
          
              # Get model output
              output = model(video_features, token_ids, teacher_forcing_ratio=0)  # Important: set teacher forcing to 0
          
              # Greedy decode the output
              output_ids = output.argmax(dim=-1)  # Shape: (batch_size, seq_len)
          
              for ref_tokens, pred_tokens in zip(token_ids, output_ids):
                  ref_sentence = [inv_vocab[idx.item()] for idx in ref_tokens if idx.item() not in {0, 1, 2}]
                  pred_sentence = [inv_vocab[idx.item()] for idx in pred_tokens if idx.item() not in {0, 1, 2}]
          
                  references.append([ref_sentence])  # BLEU expects list of references
                  hypotheses.append(pred_sentence)
          
      # Now calculate BLEU
      bleu_score = 0.0
      for ref, hyp in zip(references, hypotheses):
          bleu_score += sentence_bleu(ref, hyp, smoothing_function=smooth_fn)
      
      avg_bleu = bleu_score / len(references)
      bleu_scores[epoch] = avg_bleu 
      
      if avg_bleu > best_bleu:
          best_bleu = avg_bleu
          torch.save(model.state_dict(), "/mnt/scratch/od22kob/MSC Final/Sentry-AI/Python/Models/Best_Models/best_model_bleu.pt")
          print(f"New best BLEU {best_bleu:.4f}! Model saved.")
  
      print(f"Validation BLEU Score: {avg_bleu:.4f}")
      print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Test BLEU: {avg_bleu:.4f}")
  
  
  
  
  fig, ax1 = plt.subplots()
  
  ax1.plot(losses[0], label='Training Loss', color='blue')
  ax1.plot(losses[1], label='Testing Loss', color='orange')
  ax1.set_xlabel('Epochs')
  ax1.set_ylabel('Loss')
  ax1.legend(loc='upper left')
  ax1.grid(True)
  
  ax2 = ax1.twinx()
  ax2.plot(bleu_scores, label='Average BLEU Score', color='green')
  ax2.set_ylabel('BLEU Score')
  
  lines_1, labels_1 = ax1.get_legend_handles_labels()
  lines_2, labels_2 = ax2.get_legend_handles_labels()
  ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center')
  
  plt.title("SentryNet (SlowFast+LSTM) Captioning Model")
  plt.savefig("Training_SentryNet.png")
  plt.show()
  
# In[ ]:
  
# Generate examples ###################################################



# Load the best BLEU model before running the test set
model_path = "/mnt/scratch/od22kob/MSC Final/Sentry-AI/Python/Models/Best_Models/best_model_bleu.pt"
model.load_state_dict(torch.load(model_path))
model.eval()  # Set the model to evaluation mode

# Run through test set and calculate BLEU scores
references = []
hypotheses = []
bleu_scores_per_clip = []

# Store BLEU scores and corresponding video data for the top 3 clips
top_3_clips = []

with torch.no_grad():
    for video_features, token_ids in tqdm(test_loader, desc="Testing BLEU Score Calculation"):
        video_features = [feature.to(device) for feature in video_features]
        token_ids = token_ids.to(device)

        # Get model output
        output = model(video_features, token_ids, teacher_forcing_ratio=0)  # No teacher forcing for inference

        # Greedy decode the output
        output_ids = output.argmax(dim=-1)  # Shape: (batch_size, seq_len)

        for i, (ref_tokens, pred_tokens) in enumerate(zip(token_ids, output_ids)):
            # Convert token indices to words using vocab
            ref_sentence = [inv_vocab[idx.item()] for idx in ref_tokens if idx.item() not in {0, 1, 2}]
            pred_sentence = [inv_vocab[idx.item()] for idx in pred_tokens if idx.item() not in {0, 1, 2}]

            references.append([ref_sentence])  # BLEU expects list of references
            hypotheses.append(pred_sentence)

            # Calculate BLEU score for this clip
            bleu_score = sentence_bleu([ref_sentence], pred_sentence, smoothing_function=smooth_fn)
            bleu_scores_per_clip.append(bleu_score)

            # Store the clip details if it's in the top 3 BLEU scores
            if len(top_3_clips) < 3:
                top_3_clips.append((bleu_score, video_features[0][i], ref_sentence, pred_sentence))  # Save BLEU score, video frame, and captions
            else:
                # Check if the current BLEU score is higher than any of the top 3
                min_bleu_score = min(top_3_clips, key=lambda x: x[0])
                if bleu_score > min_bleu_score[0]:
                    top_3_clips.remove(min_bleu_score)
                    top_3_clips.append((bleu_score, video_features[0][i], ref_sentence, pred_sentence))

# Sort top 3 clips by BLEU score
top_3_clips.sort(reverse=True, key=lambda x: x[0])

# Display results for the top 3 clips
for i, (bleu_score, frame, ref_caption, gen_caption) in enumerate(top_3_clips):
    print(f"Top {i+1} Clip (BLEU: {bleu_score:.4f}):")
    print(f"Original Caption: {' '.join(ref_caption)}")
    print(f"Generated Caption: {' '.join(gen_caption)}")
    
    # Display first frame (assuming `frame` is a tensor of shape [C, H, W])
    print(f"Frame shape before conversion: {frame.shape}")
    
    # Extract the first frame and convert it to numpy array
    first_frame = frame[0].cpu().numpy().transpose(1, 2, 0)  # Convert to [H, W, C]
    
    plt.imshow(first_frame)
    plt.axis('off')
    plt.title(f"First Frame of Clip {i+1}")
    plt.show()