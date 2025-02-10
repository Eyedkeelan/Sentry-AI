import os
import cv2 as cv
import numpy as np
import pandas as pd 
import torch
from torch import nn
from torchvision import models, transforms
from tqdm import tqdm


VIDEO_DIR = r"C:\Users\Keelan.Butler\Desktop\python_projects\Final Project\OneDrive_2025-01-30\MSAD Dataset\MSAD_blur"

OUTPUT_DIR = r"C:\Users\Keelan.Butler\Desktop\python_projects\Final Project\Dataset" 

def extract_directory(VIDEO_DIR = VIDEO_DIR, OUTPUT_DIR= OUTPUT_DIR):
    FRAME_INTERVAL = 5

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "frames"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "features"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = models.resnet18(pretrained = True)
    model = nn.Sequential(*list(model.children())[:-1]) # Removing the final layer enables me to extract features as opposed to a final layer

    transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224,224)),transforms.ToTensor()])


    # Process videos recursively
    metadata = []
    for root, _, files in os.walk(VIDEO_DIR):
        for video_file in tqdm(files):
            if not video_file.endswith((".mp4", ".avi", ".mov")):
                continue

            video_path = os.path.join(root, video_file)
            relative_path = os.path.relpath(root, VIDEO_DIR)  # Preserve subfolder structure

            # Create output folders for frames and features
            frame_output_dir = os.path.join(OUTPUT_DIR, "frames", relative_path)
            feature_output_dir = os.path.join(OUTPUT_DIR, "features", relative_path)
            os.makedirs(frame_output_dir, exist_ok=True)
            os.makedirs(feature_output_dir, exist_ok=True)

            cap = cv.VideoCapture(video_path)
            frame_count = 0

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                if frame_count % FRAME_INTERVAL == 0:
                    # Save frame
                    frame_filename = f"{video_file}_frame_{frame_count}.jpg"
                    frame_path = os.path.join(frame_output_dir, frame_filename)
                    cv.imwrite(frame_path, frame)

                    # Feature extraction
                    image_tensor = transform(frame).unsqueeze(0).to(device)
                    with torch.no_grad():
                        features = model(image_tensor)
                    feature_vector = features.cpu().numpy().flatten()

                    # Save feature vector
                    feature_path = os.path.join(feature_output_dir, frame_filename.replace(".jpg", ".npy"))
                    np.save(feature_path, feature_vector)

                    # Store metadata
                    metadata.append([video_file, frame_count, frame_path, feature_path, relative_path])

                frame_count += 1
            cap.release()

    # Save metadata to CSV
    df = pd.DataFrame(metadata, columns=["video_file", "frame_number", "frame_path", "feature_path", "subfolder"])
    df.to_csv(os.path.join(OUTPUT_DIR, "metadata.csv"), index=False)

    print("Processing complete. Frames and features saved.")

if __name__ == "__main__": 

    extract_directory()


