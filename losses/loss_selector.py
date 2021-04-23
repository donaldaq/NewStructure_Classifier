import torch.nn as nn

def loss_selector(lossName):
    if lossName == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
    elif lossName == "BCEWithLogitsLoss":
        criterion = nn.BCEWithLogitsLoss()
    else:
        None
    
    return criterion