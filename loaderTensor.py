# -- coding: utf-8 --
"""
Created on Sat Jan 11 13:13:24 2020

@author: basit
"""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import random

class ImageDataset(Dataset):
    """MIT image dataset."""

    def __init__(self, root_dir, trainPercent, valPercent, flag, transform=None):
        self.root_dir  = root_dir
#        self.dataPtr   = sorted(os.listdir(root+"images/"), key=len)
        self.transform = transform
#        self.flag = flag
        with open(self.root_dir + "cleanTrainCap", "rb") as f:
            trainCaps = pickle.load(f)
        with open("cleanTrainID", "rb") as f:
            trainID = pickle.load(f) - 1
        random.seed(35)
        randInd = np.arange(len(trainID))
        random.shuffle(randInd)
        trainCaps = trainCaps[randInd]
        trainID = trainID[randInd]
#        np.random.shuffle(trainID)
        lengths = [int(len(trainID)*trainPercent), int(len(trainID)*(trainPercent+valPercent))]#, (len(self.trainID) - (int(len(self.trainID)*trainPercent)+ int(len(self.trainID)*valPercent)))]
#        train = trainID[:lengths[0]]
#        val   = trainID[lengths[0]:lengths[1]]
#        test  = trainID[lengths[1]:]
        if flag == 'train':
            self.imgID = trainID[:lengths[0]]
            self.captions = trainCaps[:lengths[0]]
        elif flag == 'validation':
            self.imgID = trainID[lengths[0]:lengths[1]]
            self.captions = trainCaps[lengths[0]:lengths[1]]
        elif flag == 'test':
            self.imgID = trainID[lengths[1]:]
            self.captions = trainCaps[lengths[1]:]

    def __len__(self):
        return len(self.imgID)

    def __getitem__(self, idx):
        imgName = self.imgID[idx]
        img_name = self.root_dir + "features2/featuretensor(" + str(imgName) + ").pt"
        image = torch.load(img_name)
        # image = Image.open(img_name)
        # if image.mode == 'L':
        #     image = image.convert('RGB')
        # elif image.mode == 'CMYK':
        #     image = image.convert('RGB')  
#        if image.size[0] == 1:
#            image = image.repeat(3,1,1)
#        elif  image.size[0] == 4:
#            image = image[0].repeat(3,1,1)
        # if self.transform:
        #     image = self.transform(image)
        
        caption = self.captions[idx]

        sample = {'image': image, 'caption': torch.from_numpy(np.array(caption))}
        return sample