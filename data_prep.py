import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    """MIT image dataset."""

    def __init__(self, root_dir, data, transform = None):
        self.root_dir  = root_dir
        self.dataPtr   = data
        self.transform = transform
        with open(self.root_dir + "cleanTrainCap", "rb") as f:
            self.trainCap = pickle.load(f)
        with open("cleanTrainID", "rb") as f:
            self.trainID = pickle.load(f) - 1

    def __len__(self):
        return len(self.dataPtr)

    def __getitem__(self, idx):
        img_name = self.root_dir + "images/" + self.dataPtr[idx]
        image = Image.open(img_name)
        if image.mode == 'L':
            image = image.convert('RGB')
        elif image.mode == 'CMYK':
            image = image.convert('RGB')
        
#        if image.size[0] == 1:
#            image = image.repeat(3,1,1)
#
#        elif  image.size[0] == 4:
#            image = image[0].repeat(3,1,1)
            
        if self.transform:
            image = self.transform(image)

        imgNum = int(self.dataPtr[idx][3:-4])
        # captions = self.trainCap[self.trainID == imgNum]
        # rand    = np.random.randint(captions.shape[0])
        # caption = captions[rand]

        # sample = {'image': image, 'caption': torch.from_numpy(np.array(caption))}
        sample = image
        return sample, imgNum