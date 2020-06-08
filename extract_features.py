# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 18:28:10 2020

@author: canbe
"""

import torch
import os
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms
from PIL import Image
import pandas as pd
from codec import EncoderCNN, DecoderRNN
from data_prep import ImageDataset
import numpy as np
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root = "C:/pr_files/neural/"

totalList = sorted(os.listdir(root+"images/"), key=len)

dataset_tr   = ImageDataset(root_dir = root, data = totalList, transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.Resize((299,299),interpolation=Image.NEAREST),
                                                                                           transforms.ToTensor()]))
dataloader_train = DataLoader(dataset_tr, batch_size=16, shuffle=False)
encoder = EncoderCNN().to(device)
for i, (data, name) in enumerate(dataloader_train):
    
    imgFeat = data.to(device)
    # print(data.size())
    features = encoder(imgFeat)
    # img1 = features[0]
    # img2 = features[1]
    for j in range(16):
        torch.save(features[j], 'C:/pr_files/features2/feature'+ str(name[j]) + '.pt')
        # torch.save(data[j], 'C:/pr_files/features/feature'+ name[1] + '.pt')
    if i % 100 == 0:
        print(i)
        # torch.load('file.pt')