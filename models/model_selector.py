import torch
import torch.nn as nn
from torchvision import models
# pretrained models import
import pretrainedmodels

import timm
# pretrained EfficientNet import
from efficientnet_pytorch import EfficientNet

def model_selector(modelName, class_names, use_fixed, device):
    """ Set Deep Learning Model

    Args:
        modelName (String): Model Name
        class_names (String): Class Name
        use_fixed (Boolean): Select fixed weight in Neural Network
        device (Integer): Device Number

    Returns:
        [type]: [description]
    """
    if modelName == 'DenseNet161':
        model_ft = models.densenet161(pretrained=True)
    elif modelName == 'InceptionResNetV2':
        model_ft = pretrainedmodels.__dict__['inceptionresnetv2'](num_classes=1000, pretrained='imagenet')
    elif modelName == 'NASNetALarge':
        model_ft = pretrainedmodels.__dict__['nasnetalarge'](num_classes=1000, pretrained='imagenet')
    elif modelName == 'ResNet152':
        model_ft = models.resnet152(pretrained=True)
    elif modelName == 'EfficientNet':
        model_ft = EfficientNet.from_pretrained('efficientnet-b5')
    elif modelName == 'ResumeModel':
        model_ft = torch.load('./models/mal_7/InceptionResNetV2_10.pth')
    elif modelName == 'DenseNet169':
        model_ft = models.densenet169(pretrained=True)
    else:
        print('Check model name!')
    
    model_name = model_ft.__class__.__name__
    print('model name: '+ model_name)
    return model_settings(model_ft, model_name, use_fixed, class_names, device)


def model_settings(model_ft, model_name, use_fixed, class_names, device):
    """ Set Model Settings

    Args:
        model_ft (Object): Pretrained Model
        model_name (String): Model name
        use_fixed (Boolean): Select fixed weight in Neural Network
        class_names (String): Class name for Counting class
        device (Integer): Device Number

    Returns:
        [Object]: Pretrained Model
    """
    # Using a model pre-trained on ImageNet and replacing it's final linear layer
    if use_fixed == True:
        for param in model_ft.parameters():
            param.requires_grad = False
    else: # fine-tuning
        for param in model_ft.parameters():
            param.requires_grad = True

    if model_name == 'DenseNet':
        #model_ft.avgpool_1a = nn.AdaptiveAvgPool2d(1)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, len(class_names))
    elif model_name == 'InceptionResNetV2':
        num_ftrs = model_ft.last_linear.in_features
        model_ft.avgpool_1a = nn.AdaptiveAvgPool2d(1)
        model_ft.last_linear = nn.Linear(num_ftrs, len(class_names))
    elif model_name == 'ResNet':
        #model_ft = models.resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
        model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    elif model_name == 'VGG16_BN':
        model_ft = models.vgg16_bn(pretrained=True)
        model_ft.classifier[6].out_features = 8
    elif model_name == 'EfficientNet':
        num_ftrs = model_ft._fc.in_features
        model_ft._fc = nn.Linear(num_ftrs, len(class_names))
    elif model_name == 'NASNetALarge':
        num_ftrs = model_ft.last_linear.in_features
        model_ft.avg_pool = nn.AdaptiveAvgPool2d(1)
        model_ft.last_linear = nn.Linear(num_ftrs, len(class_names))
    
    model_ft = model_ft.to(device)

    return model_ft
