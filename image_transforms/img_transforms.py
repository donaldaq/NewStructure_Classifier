from torchvision import datasets, models, transforms
from yaml import serialize
from utils.autoaugment import ImageNetPolicy


def get_transforms(size):
    """ Get Tranforms

    Returns:
        [Dictionary]: Transforms 
    """

    # Data augmentation and normalization for training
    # Just normalization for validation & test

    data_transforms = {
        'train': transforms.Compose([
            #transforms.RandomResizedCrop(299, scale=(0.5,1.0)),
            transforms.Resize((size,size)),
            ImageNetPolicy(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((size,size)),
            #transforms.CenterCrop(299),
            #transforms.Resize((600,600)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((size,size)),
            #transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    return data_transforms