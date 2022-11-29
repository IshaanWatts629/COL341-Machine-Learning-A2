import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from skimage import io, transform

import matplotlib.pyplot as plt # for plotting
import numpy as np
import pandas as pd
import glob
import os
from tqdm import tqdm

from IPython.display import Image
import cv2
import sys

args = sys.argv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ImageDataset(Dataset):
    
    def __init__(self, data_csv, train = True , img_transform=None):
        """
        Dataset init function
        
        INPUT:
        data_csv: Path to csv file containing [data, labels]
        train: 
            True: if the csv file has [labels,data] (Train data and Public Test Data) 
            False: if the csv file has only [data] and labels are not present.
        img_transform: List of preprocessing operations need to performed on image. 
        """
        
        self.data_csv = data_csv
        self.img_transform = img_transform
        self.is_train = train
        
        data = pd.read_csv(data_csv, header=None)
        if self.is_train:
            images = data.iloc[:,1:].to_numpy()
            labels = data.iloc[:,0].astype(int)
        else:
            images = data.iloc[:,:].to_numpy()
            labels = None
        
        self.images = images
        self.labels = labels
        print("Total Images: {}, Data Shape = {}".format(len(self.images), images.shape))
        
    def __len__(self):
        """Returns total number of samples in the dataset"""
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Loads image of the given index and performs preprocessing.
        
        INPUT: 
        idx: index of the image to be loaded.
        
        OUTPUT:
        sample: dictionary with keys images (Tensor of shape [1,C,H,W]) and labels (Tensor of labels [1]).
        """
        image = self.images[idx]
        image = np.array(image).astype(np.uint8).reshape((32,32,3), order="F")
        
        if self.is_train:
            label = self.labels[idx]
        else:
            label = -1
        
        image = self.img_transform(image)
        
        sample = {"images": image, "labels": label}
        return sample

# Data Loader Usage

BATCH_SIZE = 200 # Batch Size. Adjust accordingly
NUM_WORKERS = 20 # Number of threads to be used for image loading. Adjust accordingly.

img_transforms = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])

# Test DataLoader
test_data = args[1] # Path to test csv file
test_dataset = ImageDataset(data_csv = test_data, train=False, img_transform=img_transforms)
test_loader_new = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=False, num_workers = NUM_WORKERS)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1)
        self.norm1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,kernel_size=3, stride=1)
        self.norm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=512,kernel_size=3, stride=1)
        self.norm3 = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv4 = nn.Conv2d(in_channels=512, out_channels=1024,kernel_size=2, stride=1)
        
        self.fc1 = nn.Linear(1024, 256)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(256, 10)
        
    def forward(self, x):
        #print(f'In Shape : {x.size()}')
        
        x = F.relu(self.norm1(self.conv1(x)))
        #print(f'Post Conv1 : {x.size()}')
        x = self.pool1(x)
        #print(f'Post Pool1 : {x.size()}')
        
        x = F.relu(self.norm2(self.conv2(x)))
        #print(f'Post Conv2 : {x.size()}')
        x = self.pool2(x)
        #print(f'Post Pool1 : {x.size()}')
        
        x = F.relu(self.norm3(self.conv3(x)))
        #print(f'Post Conv3 : {x.size()}')
        x = self.pool3(x)
        #print(f'Post Pool1 : {x.size()}')
        
        x = F.relu(self.conv4(x))
        #print(f'Post Conv4 : {x.size()}')
        
        x = x.view(-1, 1024)

        x = self.dropout(F.relu(self.fc1(x)))
        #print(f'Post Fc1 : {x.size()}')
        
        x = self.fc2(x)
        #print(f'Post Fc2 : {x.size()}')

        return x


model_test = ConvNet().to(device)
model_test.load_state_dict(torch.load(args[2]))
model_test.eval()

predictions = np.array([])

for batch_idx, sample in enumerate(test_loader_new):
    images = sample['images']
    labels = sample['labels']
            
    images = images.to(device)
    labels = labels.to(device)
            
    outputs = model_test(images)

    _, pred = torch.max(outputs, 1)
    
    predictions = np.append(predictions, pred.cpu().numpy())
    
np.savetxt(args[3], predictions)