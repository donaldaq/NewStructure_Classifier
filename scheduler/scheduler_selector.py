## Selection Optimizer
from torch.optim import lr_scheduler

def scheduler_selector(model_ft,optimizer_ft,optimName):


    if optimName == "StepLR":
        ## Decay LR by a factor of 0.1 every 10 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
    elif optimName == "MultiStepLR":
        ## Decay LR by a factor of 0.1 every 10 epochs in 3 times
        exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[10,20,30], gamma=0.1)
    elif optimName == "WarmupCosineAnnealing":
        ## Warmup Cosine Anealing Scheduler
        exp_lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer_ft, T_0=50, T_mult=1, eta_min=1e-7, verbose=True)
    else:
        None

    return exp_lr_scheduler



    
        



## Using Adam as the parameter optimizer

#

### Selection schedulers





