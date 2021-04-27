import sys
import os
import os.path as path 
import warnings
warnings.filterwarnings(action='ignore')
import glob
import numpy as np

import torch
from torch.utils import data
from torchvision import datasets, models, transforms
from datetime import datetime
import image_transforms.img_transforms
from PIL import Image as PIL
from tqdm import tqdm

#import datasets
import utils.balanced_classes as balanced_classes

# ImageNet Policy
from utils.autoaugment import ImageNetPolicy


class CustomDataset(data.Dataset):
    def __init__(self, root_dir, transforms=None):
        
        def regular_ext(extension):
            return '*.' + ''.join('[%s%s]' % (e.lower(), e.upper()) for e in extension)

        self.root_dir = root_dir
        self.classes = os.listdir(self.root_dir)
        self.transforms = transforms
        self.data = []
        self.labels = []
        for idx, cls in enumerate(self.classes):
            cls_dir = os.path.join(self.root_dir, cls)

        for extname in ['png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp', 'webp']:
            reg_ext = regular_ext(extname)
            for img in glob(os.path.join(cls_dir, reg_ext)):
                self.data.append(img)
                self.labels.append(idx)


    def __getitem__(self, idx):
        img_path, label = self.data[idx], self.labels[idx]
        img = PIL.Image.open(img_path)

        if self.transforms:
            img = self.transforms(img)
        
        sample = { "image": img, "label" : label, "filename" : img_path }
        return sample
    
    def __len__(self):
        return len(self.data)