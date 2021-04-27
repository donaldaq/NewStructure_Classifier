import torch.nn as nn

def loss_selector(lossName):
    """ Set Loss

    Args:
        lossName (String): Select Loss Name

    Returns:
        [Object]: Loss
    """
    if lossName == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
    elif lossName == "BCEWithLogitsLoss":
        criterion = nn.BCEWithLogitsLoss()
    else:
        None
    
    return criterion