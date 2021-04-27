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
import image_transforms.img_transforms as image_transforms
from PIL import Image
from tqdm import tqdm

#import datasets
import utils.balanced_classes as balanced_classes

# ImageNet Policy
from utils.autoaugment import ImageNetPolicy


class ImageFolderDataSet():
    
    def __init__(self, cfg):

        self.date = datetime.today().strftime("%-y%m%d")

        self.data_dir = cfg['model']['data_dir']
        
        self.img_size = (cfg['augmentations']['train']['resize'], cfg['augmentations']['train']['resize'], 3)
        self.batch_size_control = int(cfg['hyper_params']['batch_size'])

        #transforms = img_transforms()
        transforms = image_transforms.get_transforms()

        self.image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x), transforms[x])
                    for x in ['train', 'val', 'test']}

        # dataset size and class names check 
        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in ['train', 'val', 'test']}
        self.class_names = self.image_datasets['train'].classes
        print(self.dataset_sizes)
        

    def __len__(self):
        """Return the number of images."""
        return self.dataset_sizes


