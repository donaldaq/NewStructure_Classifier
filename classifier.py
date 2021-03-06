
import warnings
warnings.filterwarnings(action='ignore')

import torch
import torch.nn as nn
import torch.optim as optim
#from torch.optim import lr_scheduler
import numpy as np
#import torchvision
from torchvision import models #, transforms
import matplotlib.pyplot as plt
import time
import os
import sys
import copy
import seaborn as sn
import pandas as pd
import torchnet.meter.confusionmeter as cm
from tqdm import tqdm
from datetime import datetime
import datasets.ImageDataLoader as ImageDataLoader
import image_transforms.img_transforms as image_transforms


# model summary
#from torchsummary import summary

# pretrained models import
#import pretrainedmodels

# pretrained EfficientNet import
#from efficientnet_pytorch import EfficientNet

# balanced batch sampler
#import utils.balanced_classes as balanced_classes

# ImageNet Policy
#from utils.autoaugment import ImageNetPolicy

# TensorBoard setup
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('log/classification')

# import EarlyStopping
from utils.pytorchtools import EarlyStopping

# model load 
import models.model_selector

# optimizer load
import optimizer.optimizer_selector

# scheduler load
import scheduler.scheduler_selector

# loss load
import losses.loss_selector

# yaml for configuration
import yaml

# Report
from report.report import make_Report

# Set Date
date = datetime.today().strftime("%-y%m%d")

# Set yaml Configuration
ymlfile = sys.argv[1]
config_path = 'config'
yml_path = os.path.join(config_path, ymlfile)

with open(yml_path, 'r') as stream:
    cfg = yaml.safe_load(stream)

result_path = 'results'
global result_file_path
result_file_path = os.path.join(result_path, '{}_{}_{}_train_results.txt'.format(date, cfg['model']['arch'], cfg['model']['save_dir']))
global result_report
result_report = open(result_file_path, 'w', encoding='utf_8')

############## Check Training Information ##############
print('-----------check summary-----------', file=result_report)
print('cuda number: ', cfg['cuda_number'], file=result_report)
print('model name: ', cfg['model']['arch'], file=result_report)
print('use pretrained model: ', cfg['model']['params']['pretrained'], file=result_report)
print('dataset: ', cfg['model']['data_dir'], file=result_report)
print('save_dir: ', cfg['model']['save_dir'], file=result_report)
print('using fixed pretrained model parameter: ', cfg['model']['fixed'], file=result_report)
print('balaced classes control: ', cfg['model']['balanced_class'], file=result_report)

print('-----------hyper parameter summary-----------', file=result_report)
print('epoch number: ', cfg['hyper_params']['epoch'], file=result_report)
print('batch size: ',cfg['hyper_params']['batch_size'], file=result_report)
print('learning rate: ', cfg['hyper_params']['optimizer']['params']['lr'], file=result_report)
print('optimizer: ', cfg['hyper_params']['optimizer']['name'], file=result_report)
print('loss: ', cfg['hyper_params']['loss']['name'], file=result_report)
print('image resize: ', cfg['augmentations']['train']['resize'], file=result_report) 

print('---------earlystop summary--------', file=result_report)
print('control: ', cfg['early_stop']['control'], file=result_report)
print('start point: ', cfg['early_stop']['startnumber'], file=result_report)
print('patince number: ', cfg['early_stop']['patiencenumber'], file=result_report)



######################### Print Setting Parameter ###########################################
print('-----------check summary-----------')
print('cuda number: ', cfg['cuda_number'])
print('model name: ', cfg['model']['arch'])
print('use pretrained model: ', cfg['model']['params']['pretrained'])
print('dataset: ', cfg['model']['data_dir'])
print('save_dir: ', cfg['model']['save_dir'])
print('using fixed pretrained model parameter: ', cfg['model']['fixed'])
print('balaced classes control: ', cfg['model']['balanced_class'])

print('-----------hyper parameter summary-----------')
print('epoch number: ', cfg['hyper_params']['epoch'])
print('batch size: ',cfg['hyper_params']['batch_size'])
print('learning rate: ', cfg['hyper_params']['optimizer']['params']['lr'])
print('optimizer: ', cfg['hyper_params']['optimizer']['name'])
print('loss: ', cfg['hyper_params']['loss']['name'])
print('image resize: ', cfg['augmentations']['train']['resize'])

print('---------earlystop summary--------')
print('control: ', cfg['early_stop']['control'])
print('start point: ', cfg['early_stop']['startnumber'])
print('patince number: ', cfg['early_stop']['patiencenumber'])

############## Check Training Information End ##############



### Selection GPU Server Numbers
os.environ["CUDA_VISIBLE_DEVICES"] = cfg['cuda_number']

### Set Data and Save_dir 
data_dir = cfg['model']['data_dir']
save_dir = cfg['model']['save_dir']

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### Set Early Stop Configuration
earlystop_control = cfg['early_stop']['control']
earlystop_startnumber = cfg['early_stop']['startnumber']
earlystop_patiencenumber = cfg['early_stop']['patiencenumber']

#use_fixed = cfg['model']['fixed']
#lr = float(cfg['hyper_params']['optimizer']['params']['lr'])

### Declaration Lists For Graph Generation
epoch_counter_train = []
epoch_counter_val = []
train_loss = []
val_loss = []
train_acc = []
val_acc = []

### Cuda Device Check 
print('cuda device count check: ', torch.cuda.device_count())
print('cuda available check :', torch.cuda.is_available())
result_report.close()

#Train the model
def train_model(model, criterion, optimizer, scheduler, dataloaders, dataloaders_val, dataset_sizes, model_name, num_epochs):
    """ Model Training

    Args:
        model (Object): pretrained model
        criterion (Object): Loss 
        optimizer (Object): Optimizer
        scheduler (Object): [description]
        dataloaders (Dictionary): Train dataset Load
        dataloaders_val (Dictionary): Validation dataset Load
        dataset_sizes (Dictionary): All of dataset size(length information)
        model_name (String): Train model name
        num_epochs (Integer): Number of Epochs

    Returns:
        Object: Best Train Model(Criterion of Validation Loss)
    """
    since = time.time()
    #Initialize Values
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 1000.0
    best_epoch = 0

     # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=earlystop_patiencenumber, verbose=True)


    for epoch in range(num_epochs):
        epoch = epoch + 1
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        result_report = open(result_file_path, 'a', encoding='utf_8')

        print('Epoch {}/{}'.format(epoch, num_epochs), file= result_report)
        print('-' * 10, file=result_report)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                print("Current Learnig rates: ", scheduler.get_lr(), file= result_report)
                model.train()  # Set model to training mode
                loaders = dataloaders[phase]
            else:
                model.eval()   # Set model to evaluate mode
                loaders = dataloaders_val[phase]

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(loaders, mininterval=1):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()



                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            #For graph generation
            if phase == "train":
                train_loss.append(running_loss/dataset_sizes[phase])
                train_acc.append(running_corrects.double() / dataset_sizes[phase])
                epoch_counter_train.append(epoch)
            if phase == "val":
                val_loss.append(running_loss/ dataset_sizes[phase])
                val_acc.append(running_corrects.double() / dataset_sizes[phase])
                epoch_counter_val.append(epoch)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            #for printing
            if phase == "train":
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == "val":
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                if earlystop_control == True:
                    earlystop_epoch = epoch
                    if earlystop_epoch == earlystop_startnumber:
                        # First of all, Initiallize setting best_loss
                        early_stopping(best_loss, model)
                    elif earlystop_epoch > earlystop_startnumber:
                        # early_stopping needs the validation loss to check if it has decresed,
                        # and if it has, it will make a checkpoint of the current model
                        early_stopping(epoch_loss, model)
                        #print("if this is not best loss, do not save model, check best loss: {}".format(best_loss))



            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc), file=result_report )



            # deep copy the best model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_acc = epoch_acc
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
                if torch.cuda.device_count() == 1:
                    torch.save(model, "./learned_models/{}/{}_best.pth".format(save_dir,model_name))
                    torch.save(model.state_dict(), "./learned_models/{}/{}_best_state.pth".format(save_dir,model_name))
                else:
                    torch.save(model.module.state_dict(), "./learned_models/{}/{}_best.pth".format(save_dir,model_name))
                print("Best model epoch number check: {}".format(epoch))
            if epoch % 10 == 0:
                if torch.cuda.device_count() == 1:
                    torch.save(model, "./learned_models/{}/{}_{}.pth".format(save_dir,model_name,epoch))
                else:
                    torch.save(model.module.state_dict(), "./learned_models/{}/{}_{}.pth".format(save_dir,model_name,epoch))

        writer.add_scalar("Loss/train", epoch_loss)
        writer.add_scalar("Accuracy/train", epoch_acc)



        if earlystop_control == True:
            if phase == "val" and early_stopping.early_stop:
                print("Early stopping")
                break



    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}, epoch number: {}'.format(best_acc, best_epoch))

    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60), file=result_report)
    print('-' * 10, file=result_report)
    print('-' * 10, file=result_report)
    print('Best val Acc: {:4f}, epoch number: {}'.format(best_acc, best_epoch), file=result_report)

    result_report.close()

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model









if __name__ == "__main__":

    ### Settings
    imgdataloader = ImageDataLoader.ImageDataLoader(cfg)
    datasetsize = imgdataloader.getter_datasetsize()
    classnames = imgdataloader.getter_classnames()
    dataloaders, dataloaders_val = imgdataloader.getter_dataloaders()
    optimName = cfg['hyper_params']['optimizer']['name']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    ### Hyper Parameters
    model_ft = models.model_selector.model_selector(cfg['model']['arch'], classnames, cfg['model']['fixed'], device)
    optimizer_ft = optimizer.optimizer_selector.optimizer_selector(model_ft, cfg['hyper_params']['optimizer']['params']['lr'],optimName) 
    exp_lr_scheduler = scheduler.scheduler_selector.scheduler_selector(model_ft,optimizer_ft,cfg['hyper_params']['scheduler']['name'])
    criterion = losses.loss_selector.loss_selector(cfg['hyper_params']['loss']['name'])
    

    ### CUDA Count Check
    if torch.cuda.device_count() > 1:
        model_ft = nn.DataParallel(model_ft)
    model_Parallel = model_ft.__class__.__name__
    print('model parallel check: '+ model_Parallel)

    ### Summary model contents
    #summary(model_ft, (3, 480, 720))

    ### Training 
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, dataloaders_val, datasetsize, cfg['model']['arch'],
                       num_epochs=cfg['hyper_params']['epoch'])

    

    make_Report(save_dir, cfg['model']['arch'],epoch_counter_train,train_loss,epoch_counter_val,val_loss, train_acc,val_acc)
