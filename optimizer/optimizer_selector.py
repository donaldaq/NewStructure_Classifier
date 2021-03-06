import torch.optim as optim

def optimizer_selector(model_ft, lr, optimName):
    """ Select Optimizer

    Args:
        model_ft (Object): Pretrained Model
        lr (String): Learning rates (it should be cast to float)
        optimName (String): Optimizer Name

    Returns:
        [Object]: Optimizer
    """
    if optimName == "SGD":
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    elif optimName == "Adam":
        lr = float(lr)
        optimizer_ft = optim.Adam(model_ft.parameters(), lr = lr, betas=(0.9, 0.999))
    else:
        None


    return optimizer_ft