import os
import torch
import numpy as np
import cv2 as cv
from torchvision import transforms
from pytorchvideo.models.hub import slowfast_r50
from tqdm import tqdm

# Source files 
VIDEO_DIR = r"C:\Users\Keelan.Butler\Desktop\python_projects\Final Project\OneDrive_2025-01-30\MSAD Dataset\MSAD_blur"
FEATURES_DIR = r"C:\Users\Keelan.Butler\Desktop\python_projects\Final Project\Dataset"

slowfast_model = slowfast_r50(pretrained = True)
slowfast_model.eval()

def extract_video_features(video_path):
    """This will extract video spatiotemoral features using the SlowFast Model."""
    cap = cv.VideoCapture(video_path)
    frames = []

    transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize((224,224)), transforms.ToTensor()])


test_frames = np.load(r"C:\Users\Keelan.Butler\Desktop\python_projects\Final Project\OneDrive_2025-01-30\MSAD Dataset\MSAD-I3D-WS\MSAD-I3D-abnormal-testing\Assault_5_i3d.npy")

test_frames.shape


