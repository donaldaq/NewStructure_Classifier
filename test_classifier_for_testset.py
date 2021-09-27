import warnings
warnings.filterwarnings(action='ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import matplotlib
import time
import os
import copy
import seaborn as sn
import pandas as pd
import torchnet.meter.confusionmeter as cm
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from scipy import interp
from itertools import cycle
from datetime import datetime
from collections import OrderedDict

# pretrained models import
import pretrainedmodels

# pretrained EfficientNet import
from efficientnet_pytorch import EfficientNet

# TensorBoard setup
from torch.utils.tensorboard import SummaryWriter

# Summary
from torchsummary import summary

# Delong AUC score
<<<<<<< HEAD
import utils.delong_auc
=======
import delong_auc
>>>>>>> a78bd94aa916383a6f4066c3914cc584b33b7803
from scipy import stats

import statistics

date = datetime.today().strftime("%-y%m%d")

result_filename = './results/' + date + '_multi_result.txt'
f = open(result_filename, "w")

# Data augmentation and normalization for training
# Just normalization for validation & test
data_transforms = {
    'train': transforms.Compose([
        #transforms.Resize((testset_model_size, testset_model_size)),
        #transforms.Resize((480,720)),#Image.BICUBIC
        #transforms.RandomResizedCrop(model_size, scale=(0.5,1.0)),
        #ImageNetPolicy(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        #transforms.Resize((testset_model_size, testset_model_size)),
        #transforms.Resize((480,640)),
        #transforms.CenterCrop(model_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        #transforms.Resize((testset_model_size, testset_model_size)),
<<<<<<< HEAD
        transforms.Resize((500,500)),
=======
        #transforms.Resize((500,750)),
>>>>>>> a78bd94aa916383a6f4066c3914cc584b33b7803
        #transforms.CenterCrop(testset_model_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

<<<<<<< HEAD
data_dir = '/home/huray/workspace/data/8'
=======
data_dir = '/home/mlm08/ml/data/grp_split/eyelid_3cls_8_1113'
>>>>>>> a78bd94aa916383a6f4066c3914cc584b33b7803


image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['test']}

#weights = make_weights_for_balanced_classes(image_datasets['train'].imgs, len(image_datasets['train'].classes))
#weights = torch.DoubleTensor(weights)
#print(weights)
#sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

######################## Normal
# for mode in ['train']:
#     dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], shuffle = False,sampler=sampler, batch_size=16,
#                                               num_workers=4)
#                   for x in ['train']}
for mode in ['test']:
    dataloaders_val = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                         shuffle=True, num_workers=4)
              for x in ['test']}

for mode in ['test']:
    dataloaders_test = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                         shuffle=False, num_workers=0)
              for x in ['test']}

######################## Imbalanced Dataset Sampler
# for mode in ['train']:
#     dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16, sampler=ImbalancedDatasetSampler(dataset=image_datasets['train']),
#                                          shuffle=False, num_workers=4)
#               for x in ['train']}
# for mode in ['val','test']:
#     dataloaders1 = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
#                                          shuffle=True, num_workers=4)
#               for x in ['val','test']}
########################




dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}
class_names = image_datasets['test'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


print(image_datasets['test'])

# Manually load Saving Learning Model

<<<<<<< HEAD
#model_ft = torch.load('./models/food_8/eyelid_3cls_888_1123_InceptionResNetV2_best.pth')
model_ft = torch.load('./models/food_8/DenseNet_best.pth')
=======
model_ft = torch.load('./models/eyelid_3cls_8_1123/eyelid_3cls_888_1123_InceptionResNetV2_best.pth')
#model_ft = torch.load('./models/CIN_1280_0605/DenseNet_best_CIN_1280_77_0605.pth')
>>>>>>> a78bd94aa916383a6f4066c3914cc584b33b7803
#model_ft.avgpool_1a = nn.AdaptiveAvgPool2d(1)

#Test the accuracy with test data

correct = 0
total = 0


with torch.no_grad():
    for i, (inputs, labels) in enumerate(dataloaders_val['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model_ft(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


f.write('Accuracy of the network on the test images: %.1f %% \n' % (100 * correct / total))
print('Accuracy of the network on the test images: %.1f %%' % (
    100 * correct / total))


#Class wise testing accuracy

def class_wise_testing_accuracy():
    class_correct = list(0. for i in range(len(class_names)))
    class_total = list(0. for i in range(len(class_names)))
    class_incorrect = list(0. for i in range(len(class_names)))
    mean_tpr = np.zeros_like(class_total)
    predict_percentage = []
    label = []
    class_roc = list(0. for i in range(len(class_names)))
    with_prob = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders_val['test']):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model_ft(inputs)

                _, predicted = torch.max(outputs, 1)
                predict_percentage, label = torch.topk(outputs, 1)

    #             top_prob = predict_percentage.exp()
    #             top_prob_array = top_prob.data.cpu().numpy()[0]
    #             top_prob_array = list(map(lambda x: round(float(x), 5), top_prob_array))


                #print(dataloaders_val['test'].dataset.samples[i])

    #             print(dataloaders_val['test'].dataset.samples[i])

                point = (predicted == labels).squeeze()
                false_point = (predicted != labels).squeeze()

                for j in range(len(labels)):
                    label = labels[j]
                    class_correct[label] += point[j].item()
                    class_incorrect[label] += false_point[j].item()
                    class_total[label] += 1




    for i in range(len(class_names)):
        f.write('True Positive Accuracy of %5s : %.1f %% \n' % (class_names[i], 100 * class_correct[i] / class_total[i]) )
        print('True Positive Accuracy of %5s : %.1f %%' % (
            class_names[i], 100 * class_correct[i] / class_total[i]))
        f.write('False Positive of %5s : %.1f %% \n' % (class_names[i], 100 * class_incorrect[i] / class_total[i]))
        print('False Positive of %5s : %.1f %%' % (
            class_names[i], 100 * class_incorrect[i] / class_total[i]))


def get_output(model, loader, with_prob=True):
    y_pred, y_true, fname = [], [], []



    if with_prob:
        y_prob = []
    else:
        y_prob = None

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader,0):
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            if with_prob:
                probs = torch.nn.functional.softmax(outputs, dim=1)
            else:
                probs = None


            sample_fname, p = loader.dataset.samples[i]
            #print(sample_fname)

            fname.append(sample_fname)
            y_pred.append(preds.cpu().numpy())
            y_true.append(labels.cpu().numpy())
            if with_prob:
                y_prob.append(probs.detach().cpu().numpy())
        #print(fname)
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        if with_prob:
            y_prob = np.concatenate(y_prob)
    return y_pred, y_true, y_prob, fname


def print_roc_curve(y_test, y_score, n_classes, labels, figsize = (8, 6)):

    lw = 2
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        print('------------------------------------')
        f.write('check class{} AUC score: {} \n'.format(i, roc_auc_score(y_test[:, i], y_score[:, i])))
        print('check class{} AUC score: {} \n'.format(i, roc_auc_score(y_test[:, i], y_score[:, i])))
        print(roc_auc[i])
        print(Find_Optimal_Cutoff_Multi(y_test[:, i], y_score[:, i]))

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    fig = plt.figure(figsize=figsize)


#     plt.plot(fpr["micro"], tpr["micro"],
#             label='micro-average ROC curve (area = {0:0.2f})'
#                     ''.format(roc_auc["micro"]),
#             color='deeppink', linestyle=':', linewidth=4)


#     plt.plot(fpr["macro"], tpr["macro"],
#             label='macro-average ROC curve (area = {0:0.2f})'
#                 ''.format(roc_auc["macro"]),
#             color='navy', linestyle=':', linewidth=4)

    # linestyles = OrderedDict(
    #     [('solid', (0, ())),
    #      ('loosely dotted', (0, (1, 10))),
    #      ('dotted', (0, (1, 5))),
    #      ('densely dotted', (0, (1, 1))),
    #      ('loosely dashed', (0, (5, 10))),
    #      ('dashed', (0, (5, 5))),
    #      ('densely dashed', (0, (5, 1))),
    #      ('loosely dashdotted', (0, (3, 10, 1, 10))),
    #      ('dashdotted', (0, (3, 5, 1, 5))),
    #      ('densely dashdotted', (0, (3, 1, 1, 1))),
    #      ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
    #      ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
    #      ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])
    # colors = [['k', 'solid'], ['g', 'dotted'], ['r', 'dashed'], ['c', 'dashdot'], ['m',  linestyles['densely dashdotted']]]

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'grey', 'blue', 'green'])
    colors = cycle(['black', 'grey', 'darkgrey', 'black', 'grey', 'blue', 'green'])
    linestyles = cycle(['--', '-', '-.', ':' ])
    #labels = ['cin1','cin2','cin3','normal']
    for i, line, color in zip(range(len(class_names)), linestyles, colors):
        plt.plot(fpr[i], tpr[i], lw=lw, ls=line, color=color,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(labels[i], roc_auc[i]))

    #plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.1, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve to multi-class')
    plt.legend(loc="lower right")
    return fig

def roc_curve_main():
    # obtain outputs of the model
    #alloc_label = True
#     test_dataset = EarDataset(binary_dir=args.data_dir,
#                                    alloc_label = alloc_label,
#                                     transforms=transforms.Compose([Rescale((256, 256)), ToTensor(), Normalize()]))
#     test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    test_loader = dataloaders_val['test']
    y_pred, y_true, y_score, fname = get_output(model_ft, test_loader)
    print(y_pred.shape, y_true.shape, y_score.shape)


<<<<<<< HEAD
    labels = ['food', 'nonfood', 'pass']
=======
    labels = ['Malignancy', 'Benign', 'Normal']
>>>>>>> a78bd94aa916383a6f4066c3914cc584b33b7803

    # save the roc curve

    y_onehot = np.zeros((y_true.shape[0], len(labels)), dtype=np.uint8)
    y_onehot[np.arange(y_true.shape[0]), y_true] = 1
    sums = y_onehot.sum(axis=0)
    useless_cols = []
    for i, c in enumerate(sums):
        if c == 0:
            print('useless column {}'.format(i))
            useless_cols.append(i)

    print('Label number check: ',len(labels))

    useful_cols = np.array([i for i in range(len(labels)) if i not in useless_cols])

    y_onehot = y_onehot[:,useful_cols]
    y_score = y_score[:,useful_cols]



    fig = print_roc_curve(y_onehot, y_score, useful_cols.shape[0], labels, figsize=(8,6))
    fig.savefig(os.path.join('models', 'densenet_roc_curve.png'),dpi=300, format='png')

opt_conf = []
targetList = []


def Find_Optimal_Cutoff_Multi(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------
    list type, with optimal cutoff value

    """
    confusion_matrix = cm.ConfusionMeter(len(class_names))
#     o1 = open("o1.csv", "w")
#     o2 = open("o2.csv", "w")
#     o3 = open("o3.csv", "w")
#     o4 = open("o4.csv", "w")

    #print(fname)
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i),
                       'tpr' : pd.Series(tpr), 'fpr' : pd.Series(fpr) })
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    print(roc_t)


    y_score_opt = np.where(predicted > list(roc_t['threshold'])[0], 1, 0)

    from sklearn.metrics import confusion_matrix
    opt_conf = confusion_matrix(target, y_score_opt)


#     con_m = opt_conf
#     df_con_m = pd.DataFrame(con_m, index= [i for i in class_names], columns = [i for i in class_names])
#     sn.set(font_scale= 1.1)
#     sn.heatmap(df_con_m, annot=True,fmt='g' ,  annot_kws={"size" : 10}, cbar = False, cmap="Blues")

    ##################### 결과 파일명 뽑기
    sourceNDArray = np.array(y_score_opt)
    optList = sourceNDArray.tolist()

    targetNDArray = np.array(target)
    tarList = targetNDArray.tolist()




#     for i in range(len(optList)):
#         filename = os.path.basename(fname[i])
#         pathname = os.path.dirname(fname[i])
#         patientname = filename.split(' ')[0]

#         if tarList[i] == 0 and tarList[i] == optList[i]:
#             o1.write("{}, {}, {}, label: {}, predicted: {}, 1\n".format(pathname, filename, patientname, tarList[i], optList[i]))
#         elif tarList[i] == 0 and tarList[i] != optList[i]:
#             o2.write("{}, {}, {}, label: {}, predicted: {}, 2\n".format(pathname, filename, patientname, tarList[i], optList[i]))
#         elif tarList[i] == 1 and tarList[i] == optList[i]:
#             o4.write("{}, {}, {}, label: {}, predicted: {}, 4\n".format(pathname, filename, patientname, tarList[i], optList[i]))
#         elif tarList[i] == 1 and tarList[i] != optList[i]:
#             o3.write("{}, {}, {}, label: {}, predicted: {}, 3\n".format(pathname, filename, patientname, tarList[i], optList[i]))

#         if tarList[i] == 1 and tarList[i] != optList[i]:
#             o1.write("{}, {}, label: {}, predicted: {}, 1\n".format(filename, patientname, tarList[i], optList[i]))
# #         elif tarList[i] == 0 and tarList[i] != optList[i]:
# #             o2.write("{}, {}, label: {}, predicted: {}, 2\n".format(filename, patientname, tarList[i], optList[i]))
# #         elif tarList[i] == 1 and tarList[i] == optList[i]:
# #             o4.write("{}, {}, label: {}, predicted: {}, 4\n".format(filename, patientname, tarList[i], optList[i]))
# #         elif tarList[i] == 1 and labels != optList[i]:
# #             o3.write("{}, {}, label: {}, predicted: {}, 3\n".format(filename, patientname, tarList[i], optList[i]))

#     o1.close()
#     o2.close()
#     o3.close()
#     o4.close()
    opt_acc = str(accuracy_score(target, y_score_opt))
    f.write('------------------------------------ \n')
    f.write('optimal accuracy : '+ opt_acc+'\n')
    print('optimal accuracy : \n', accuracy_score(target, y_score_opt))

    opt_mat = str(confusion_matrix(target, y_score_opt))
    f.write('optimal conf mat : '+ opt_mat + '\n')
    print('optimal conf mat : \n', confusion_matrix(target, y_score_opt))

    tn, fp, fn, tp = confusion_matrix(target, y_score_opt).ravel() # sensitivty / specificity / ppv / npv
    # print(tn, fp, fn, tp)
    str_sensitivity = str(tp / (tp + fn))
    f.write('sensitivity : '+ str_sensitivity + '\n')
    print('sensitivity : ', tp / (tp + fn))

    str_specificity = str(tn / (fp + tn))
    f.write('specificity : ' + str_specificity + '\n')
    print('specificity : ', tn / (fp + tn))

    str_ppv = str(tp / (tp + fp))
    f.write('PPV : '+ str_ppv + '\n')
    print('PPV : ', tp / (tp + fp))

    str_npv = str(tn / (tn + fn))
    f.write('NPV : ' + str_npv + '\n')
    print('NPV : ', tn / (tn + fn))

    str_opt_thr = str(roc_t['threshold'])
    f.write('optimal threshold: '+ str_opt_thr + '\n')
    print('optimal threshold: ', roc_t['threshold'])

    f.write('------------------------------------ \n')
    print('------------------------------------')



    return list(roc_t['threshold'])

def getter_confusion_matrix():
    #Get the confusion matrix for testing data
    confusion_matrix = cm.ConfusionMeter(len(class_names))
    roc_score = 0
    data = []
    number = 0
    # a = open("1.csv", "w")
    # b = open("2.csv", "w")
    # c = open("3.csv", "w")
    # d = open("4.csv", "w")


    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders_val['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model_ft(inputs)

            _, predicted = torch.max(outputs, 1)

            #print(predicted)
            #print(predicted.item())
            #print(dataloaders_val['test'].dataset.samples[i])

            sample_fname, lb = dataloaders_test['test'].dataset.samples[i]
    #         print(sample_fname)
    #         print(lb)
            filename = os.path.basename(sample_fname)
            patientname = filename.split(' ')[0]
            #print(filename)
            #print(filename.split(' ')[0])



    #         if labels == 0 and labels == predicted.item():
    #             a.write("{}, {}, label: {}, predicted: {}, 1\n".format(filename, patientname, labels.item(), predicted.item()))
    #         elif labels == 0 and labels != predicted.item():
    #             b.write("{}, {}, label: {}, predicted: {}, 2\n".format(filename, patientname, labels.item(), predicted.item()))
    #         elif labels == 1 and labels == predicted.item():
    #             d.write("{}, {}, label: {}, predicted: {}, 4\n".format(filename, patientname, labels.item(), predicted.item()))
    #         elif labels == 1 and labels != predicted.item():
    #             c.write("{}, {}, label: {}, predicted: {}, 3\n".format(filename, patientname, labels.item(), predicted.item()))




            confusion_matrix.add(predicted, labels)
            #torch.topk(nn.Softmax(dim=1)(model(img_tensor.cuda())), 1)


        print(confusion_matrix.conf)

        return confusion_matrix.conf
    # a.close()
    # b.close()
    # c.close()
    # d.close()

def draw_heatmap(con_mat):
    #Confusion matrix as a heatmap
    con_m = con_mat
    class_names = image_datasets['test'].classes
    df_con_m = pd.DataFrame(con_m, index= [i for i in class_names], columns = [i for i in class_names])

    cm1 = con_m

    cm_sum = np.sum(cm1, axis=1, keepdims=True)
    cm_perc = cm1 / cm_sum.astype(float) * 100
    annot = np.empty_like(cm1).astype(str)
    nrows, ncols = cm1.shape
<<<<<<< HEAD
    class_names = ['food','nonfood', 'pass']
=======
    class_names = ['Malignancy','Benign', 'Normal']
>>>>>>> a78bd94aa916383a6f4066c3914cc584b33b7803

    for i in range(nrows):
        for j in range(ncols):
            c = cm1[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%d (%d%%)' % (c, p)
            elif c == 0:
                annot[i, j] = '%d (%d%%)' % (c, p)
            else:
                annot[i, j] = '%d (%d%%)' % (c, p)

    cm1 = pd.DataFrame(cm1, index= [i for i in class_names], columns = [i for i in class_names])

    plt.subplots(figsize=(11,9))
    sn.set(font_scale= 1.0)
    #sn.heatmap(df_con_m, annot=True,fmt='g' ,  annot_kws={"size" : 10}, cbar = False, cmap="Blues")
    sn.heatmap((cm1.T / cm1.sum(axis=1)).T, annot=annot, fmt='', annot_kws={"size" : 13}, cmap='Blues',cbar = True, vmin=0, vmax=1)
    plt.yticks(rotation = 0)
    plt.xticks(rotation = 0)
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.savefig(os.path.join('models', 'inceptionresnet_confusion.png'),dpi=300, format='png')


if __name__ == "__main__":


    class_wise_testing_accuracy()
    roc_curve_main()

    draw_heatmap(getter_confusion_matrix())

    f.close()

    print("Yey Test Done!!")
