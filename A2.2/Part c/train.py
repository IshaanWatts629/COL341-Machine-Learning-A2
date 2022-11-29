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
import time

start = time.time()

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

img_transforms = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

# Train DataLoader
train_data = args[1] # Path to train csv file
train_dataset = ImageDataset(data_csv = train_data, train=True, img_transform=img_transforms)
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=False, num_workers = NUM_WORKERS)

# Test DataLoader
test_data = args[2] # Path to test csv file
test_dataset = ImageDataset(data_csv = test_data, train=True, img_transform=img_transforms)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=False, num_workers = NUM_WORKERS)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(32)

        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128,kernel_size=3, stride=1, padding=1)
        self.norm3 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256,kernel_size=3, stride=1, padding=1)
        self.norm5 = nn.BatchNorm2d(256)

        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=256,kernel_size=3, stride=1, padding=1)
        self.norm7 = nn.BatchNorm2d(256)

        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout2d(p=0.05)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.dropout2(self.maxpool(x))
        
        x = F.relu(self.norm3(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = self.dropout2(self.maxpool(x))
        
        x = F.relu(self.norm5(self.conv5(x)))
        x = F.relu(self.conv6(x))
        x = self.dropout2(self.maxpool(x))

        x = F.relu(self.norm7(self.conv7(x)))
        x = F.relu(self.conv8(x))
        x = self.dropout2(self.maxpool(x))
        
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(self.dropout1(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(self.dropout1(x))
        
        return x

torch.manual_seed(51)
model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 0.0003)

n_total_steps = len(train_loader)

train_loss = []
test_acc = []
epochs = 0

while (time.time()-start) < 1740:
    batch_loss = []
    model.train()
    for batch_idx, sample in enumerate(train_loader):
        images = sample['images']
        labels = sample['labels']
        
        images = images.to(device)
        labels = labels.to(device)
    
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_loss.append(loss.item())
        
        #if (batch_idx+1)%50 == 0:
         #   print(f'epoch {epoch+1}/{num_epochs}, steps {batch_idx+1}/{n_total_steps}, loss = {loss.item():.4f}')
    
    train_loss.append(np.mean(batch_loss))
    
    with torch.no_grad():
        model.eval()
        n_correct = 0
        n_samples = 0

        for batch_idx, sample in enumerate(test_loader):
            images = sample['images']
            labels = sample['labels']
            
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)

            # value, index
            _, predictions = torch.max(outputs, 1)
            n_samples += labels.shape[0]
            n_correct += (predictions==labels).sum().item()

        acc = n_correct / n_samples
        test_acc.append(acc)

        if acc >= 0.85:
            torch.save(model.state_dict(), args[3])
            np.savetxt(args[4], train_loss)
            np.savetxt(args[5], test_acc)
            #print("Saving")

    epochs += 1

    if epochs == 50:
        break

if max(test_acc) < 0.85:
    torch.save(model.state_dict(), args[3])
    np.savetxt(args[4], train_loss)
    np.savetxt(args[5], test_acc)
    #print("Final Saving")

