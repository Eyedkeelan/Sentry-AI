import os
import cv2 as cv
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import models, transforms
from tqdm import tqdm
from collections import deque  # For maintaining a sliding window of frames

# Paths
VIDEO_DIR = r"C:\Users\Keelan.Butler\Desktop\python_projects\Final Project\OneDrive_2025-01-30\MSAD Dataset\MSAD_blur"
OUTPUT_DIR = r"C:\Users\Keelan.Butler\Desktop\python_projects\Final Project\Dataset"

# Constants
FRAME_INTERVAL = 5  # Capture every 5th frame
CLIP_LENGTH = 16  # Number of frames per clip for 3D CNN
FRAME_HEIGHT, FRAME_WIDTH = 112, 112  # r3d_18 with input 112x112

def extract_directory(VIDEO_DIR=VIDEO_DIR, OUTPUT_DIR=OUTPUT_DIR):
    """ Extracts frames from videos, stacks them into clips, and extracts 3D CNN features. """
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "clips"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "features"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Pretrained 3D CNN (Feature Extractor)
    model = models.video.r3d_18(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])  # Remove final FC layer
    model = model.to(device)
    model.eval()  # Set to evaluation mode

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((FRAME_HEIGHT, FRAME_WIDTH)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    metadata = []

    # Process videos
    for root, _, files in os.walk(VIDEO_DIR):
        for video_file in tqdm(files, desc=f"Processing: {root}"):
            if not video_file.endswith((".mp4", ".avi", ".mov")):
                continue

            video_path = os.path.join(root, video_file)
            relative_path = os.path.relpath(root, VIDEO_DIR)

            # Output directories
            clip_output_dir = os.path.join(OUTPUT_DIR, "clips", relative_path)
            feature_output_dir = os.path.join(OUTPUT_DIR, "features", relative_path)
            os.makedirs(clip_output_dir, exist_ok=True)
            os.makedirs(feature_output_dir, exist_ok=True)

            cap = cv.VideoCapture(video_path)
            frame_queue = deque(maxlen=CLIP_LENGTH)  # Sliding window to hold CLIP_LENGTH frames
            frame_count = 0
            clip_index = 0

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                if frame_count % FRAME_INTERVAL == 0:
                    frame = transform(frame)  # Transform frame
                    frame_queue.append(frame)  # Add to queue

                    if len(frame_queue) == CLIP_LENGTH:  # Ensure we have a full sequence
                        clip_tensor = torch.stack(list(frame_queue), dim=1)  # Shape: (3, T, H, W)
                        clip_tensor = clip_tensor.unsqueeze(0).to(device)  # Add batch dimension: (1, 3, T, H, W)

                        # Extract features
                        with torch.no_grad():
                            features = model(clip_tensor)
                        feature_vector = features.squeeze().cpu().numpy()  # Flatten output

                        # Save clip & features
                        clip_filename = f"{video_file}_clip_{clip_index}.npy"
                        feature_filename = f"{video_file}_clip_{clip_index}_features.npy"
                        #np.save(os.path.join(clip_output_dir, clip_filename), clip_tensor.cpu().numpy()) # As this line saves the clip also, it's advised to disable due to data storage issues
                        np.save(os.path.join(feature_output_dir, feature_filename), feature_vector)

                        # Store metadata
                        metadata.append([video_file, clip_index, clip_filename, feature_filename, relative_path])

                        clip_index += 1  # Increment clip counter

                frame_count += 1
            cap.release()

    # Save metadata
    df = pd.DataFrame(metadata, columns=["video_file", "clip_index", "clip_path", "feature_path", "subfolder"])
    df.to_csv(os.path.join(OUTPUT_DIR, "metadata.csv"), index=False)

    print("Processing complete. Clips and features saved.")

if __name__ == "__main__":
    extract_directory()
