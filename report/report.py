import time
from datetime import datetime
import matplotlib.pyplot as plt

def make_Report(save_dir, model_name, epoch_counter_train, train_loss, epoch_counter_val, val_loss, train_acc, val_acc):

    """ Make Train Graph
    """

    # Set Date
    date = datetime.today().strftime("%-y%m%d")
    
    now = time.gmtime(time.time())
    currentTime = now.tm_year + now.tm_mon + now.tm_mday

    #Plot the accuracies in train & validation

    #minposs = val_loss.index(min(val_loss))+1
    #plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

    plt.figure(1)
    plt.title("Training Vs Validation Losses")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epoch_counter_train,train_loss,color = 'r', label="Training Loss")
    plt.plot(epoch_counter_val,val_loss,color = 'g', label="Validation Loss")
    plt.legend()
    plt.savefig("./learned_models/{}/{}_Loss_{}_{}.png".format(save_dir,date,model_name,currentTime),bbox_inces='tight',pad_inches=0,dpi=100)
    #plt.savefig('/home/mlm08/ml/data/SUBMUCOSAL/Submucosal_6/test/gradcam/{}/{}_{}_{}'.format(labelfolder, labelfolder, get_class_name(c),img_name),bbox_inces='tight',pad_inches=0,dpi=100)


    #Plot the accuracies in train & validation
    plt.figure(2)
    plt.title("Training Vs Validation Accuracies")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epoch_counter_train,train_acc,color = 'r', label="Training Accuracy")
    plt.plot(epoch_counter_val,val_acc,color = 'g', label="Validation Accuracy")
    plt.legend()
    plt.savefig("./learned_models/{}/{}_Acc_{}_{}.png".format(save_dir,date,model_name,currentTime),bbox_inces='tight',pad_inches=0,dpi=100)
    #plt.show()