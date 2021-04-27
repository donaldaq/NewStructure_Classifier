import sys
import os
import os.path as path 
import warnings
warnings.filterwarnings(action='ignore')
import glob
import numpy as np

import torch
from torch.utils import data
from torchvision import models, transforms
from datetime import datetime
#import transforms.img_transforms
from PIL import Image
from tqdm import tqdm

import datasets.datasets as datasets
import datasets.custom_dataset as customdatasets

import utils.balanced_classes as balanced_classes_weight


class ImageDataLoader():
    
    def __init__(self, cfg):
        """ ImageDataset Loader Initialization 

        Args:
            cfg (Object): Configuration of Yaml for Learning
        """
        self.date = datetime.today().strftime("%-y%m%d")
        self.batch_size = int(cfg['hyper_params']['batch_size'])
 
        if cfg['dataset'] == "imagefolder":
            self.image_datasets = datasets.ImageFolderDataSet(cfg)
        elif cfg['dataset'] == "customdataset":
            self.image_datasets = customdatasets.CustomDataset()
        # Called dataset class

        if cfg['model']['balanced_class'] == True:
            #setting each class weight
            self.weights = balanced_classes_weight.make_weights_for_balanced_classes(self.image_datasets['train'].imgs, len(self.image_datasets['train'].classes))
            self.weights = torch.DoubleTensor(self.weights)
            self.sampler = data.sampler.WeightedRandomSampler(self.weights, len(self.weights))
            
            for mode in ['train']:
                self.dataloaders = {x: data.DataLoader(self.image_datasets[x], shuffle = False,sampler=self.sampler, batch_size=self.batch_size,
                                                    num_workers=4)
                            for x in ['train']}
        else:
            for mode in ['train']:
                self.dataloaders = {x: data.DataLoader(self.image_datasets.image_datasets[x], shuffle = True, batch_size=self.batch_size,
                                                    num_workers=4)
                            for x in ['train']}

        for mode in ['val','test']:
            self.dataloaders_val = {x: data.DataLoader(self.image_datasets.image_datasets[x], batch_size=self.batch_size,
                                                shuffle=True, num_workers=4)
                    for x in ['val','test']}


        self.dataset_sizes = {x: len(self.image_datasets.image_datasets[x]) for x in ['train', 'val', 'test']}
        self.class_names = self.image_datasets.image_datasets['train'].classes



    def __len__(self):
        """Return the number of images."""
        return len(self.dataset_sizes)

    def getter_datasetsize(self):
        """Return Dataset sizes."""
        return self.dataset_sizes
    
    def getter_classnames(self):
        """Return class names."""
        return self.class_names

    def getter_dataloaders(self):
        """Return Data Loaders"""
        return self.dataloaders, self.dataloaders_val
