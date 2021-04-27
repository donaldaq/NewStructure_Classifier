import warnings
warnings.filterwarnings(action='ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.multiprocessing as mp
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from PIL import ImageFile
import matplotlib.pyplot as plt
# import matplotlib.font_manager as font_manager
import matplotlib
import time
import os
import shutil
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
#from netvlad import NetVLAD

# pretrained models import
import pretrainedmodels

# pretrained EfficientNet import
#from efficientnet_pytorch import EfficientNet

# TensorBoard setup
from torch.utils.tensorboard import SummaryWriter

# Summary
#from torchsummary import summary

# Delong AUC score
#import utils.delong_auc as delong_auc
from scipy import stats

import statistics

ImageFile.LOAD_TRUNCATED_IMAGES = True
date = datetime.today().strftime("%-y%m%d")

result_filename = './results/' + date + '_binary_result.txt'
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
        transforms.Resize((224,224)),
        #transforms.CenterCrop(testset_model_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_dir = '/home/jovyan/data/food-dataset'
#data_dir = '/home/mlm08/ml/data/grp_split/cin_binary_1109/cin_binary_7'
dirnamefordraw = data_dir.split('/')[-1]


image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['test']}
# print(len(image_datasets['test']))
# test1, test2, test3 = torch.utils.data.random_split(range(len(image_datasets['test'])), [10000, 10000, 4741806], generator=torch.Generator().manual_seed(42))
# print(len(test1))

# dataloaders_test1 = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
#                                          shuffle=False, num_workers=16)
#               for x in ['test']}                                                                    

for mode in ['test']:
    dataloaders_test = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                         shuffle=False, num_workers=16)
              for x in ['test']}

for mode in ['test']:
    dataloaders_test_batch = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=2048,
                                         shuffle=False, num_workers=4)
              for x in ['test']}

#dataset_sizes = {x: len(image_datasets[x]) for x in test1}
#print('check length',dataset_sizes)
class_names = image_datasets['test'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

fig_vertical = 20
fig_horizontal = 30

# Manually load Saving Learning Model
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#torch.cuda.set_device(0)

read_tensor = transforms.Compose([
    lambda x: Image.open(x),
    lambda x: x.convert('RGB'),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
    lambda x: torch.unsqueeze(x, 0)
])

def overall_accuracy(modelselection):

    #model_ft = model_ft.load_state_dict(torch.load('./DataParallel_best.pth', map_location={'cuda:0':'cpu'}))
    #Test the accuracy with test data
    correct = 0
    total = 0

    correct2 = 0
    total2 = 0

    if modelselection == 2:
        model_ft = torch.load('./models/eyelid_bin_8_1125/eyelid_bin_8_1125_DenseNet_best.pth')
        model_ft2 = torch.load('./models/eyelid_bin_8_1125/eyelid_bin_888_1125_InceptionResNetV2_best.pth')

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloaders_test['test']):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model_ft(inputs)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloaders_test['test']):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model_ft2(inputs)

                _, predicted = torch.max(outputs.data, 1)
                total2 += labels.size(0)
                correct2 += (predicted == labels).sum().item()

        f.write('Accuracy of the network on the test images for model_ft: %.1f %% \n' % (100 * correct / total))
        print('Accuracy of the network on the test images for model_ft: %.1f %%' % (
            100 * correct / total))
        f.write('Accuracy of the network on the test images for model_ft2: %.1f %% \n' % (100 * correct2 / total2))
        print('Accuracy of the network on the test images for model_ft2: %.1f %%' % (
            100 * correct2 / total2))

        #class_wise_tesing_accuracy(model_ft)
        #class_wise_tesing_accuracy(model_ft2)
        binary_roc_curve_main_two(model_ft, model_ft2)
    elif modelselection == 1:
        model_ft = torch.load('./DenseNet_best_defaultsize.pth') #DenseNet, InceptionResNetV2
        #model_ft = torch.load('./models/cin_binary_1110/cin_bin_7_DenseNet_best.pth') #InceptionResNetV2

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloaders_test['test']):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model_ft(inputs)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        f.write('Accuracy of the network on the test images for model_ft: %.1f %% \n' % (100 * correct / total))
        print('Accuracy of the network on the test images for model_ft: %.1f %%' % (100 * correct / total))

        class_wise_tesing_accuracy(model_ft)
        binary_roc_curve_main(model_ft)
        draw_heatmap(getter_confusion_matrix(model_ft))
    elif modelselection == 3:
        print("Copying Starts!!!!")
        model_ft = torch.load('./DenseNet_best_defaultsize.pth') #DenseNet, InceptionResNetV2

        toCopy_prediction(model_ft)






def class_wise_tesing_accuracy(model_ft):
    #Class wise testing accuracy
    class_correct = list(0. for i in range(len(class_names)))
    class_total = list(0. for i in range(len(class_names)))
    class_incorrect = list(0. for i in range(len(class_names)))
    mean_tpr = np.zeros_like(class_total)
    predict_percentage = []
    label = []
    class_roc = list(0. for i in range(len(class_names)))
    with_prob = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders_test_batch['test']):
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


# def class_wise_tesing_accuracy(model_ft):
#     #Class wise testing accuracy
#     class_correct = list(0. for i in range(len(class_names)))
#     class_total = list(0. for i in range(len(class_names)))
#     class_incorrect = list(0. for i in range(len(class_names)))
#     mean_tpr = np.zeros_like(class_total)
#     predict_percentage = []
#     label = []
#     class_roc = list(0. for i in range(len(class_names)))
#     with_prob = []
#     with torch.no_grad():
#         for i, (inputs, labels) in enumerate(dataloaders_test_batch['test']):
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)
#                 outputs = model_ft(inputs)

#                 _, predicted = torch.max(outputs, 1)
#                 predict_percentage, label = torch.topk(outputs, 1)

#     #             top_prob = predict_percentage.exp()
#     #             top_prob_array = top_prob.data.cpu().numpy()[0]
#     #             top_prob_array = list(map(lambda x: round(float(x), 5), top_prob_array))


#                 #print(dataloaders_val['test'].dataset.samples[i])

#     #             print(dataloaders_val['test'].dataset.samples[i])

#                 point = (predicted == labels).squeeze()
#                 false_point = (predicted != labels).squeeze()

#                 for j in range(len(labels)):
#                     print(point)
#                     label = labels[j]
#                     class_correct[label] += point[j].item()
#                     class_incorrect[label] += false_point[j].item()
#                     class_total[label] += 1




    # for i in range(len(class_names)):
    #     #f.write('True Positive Accuracy of %5s : %.1f %% \n' % (class_names[i], 100 * class_correct[i] / class_total[i]) )
    #     print('True Positive Accuracy of %5s : %.1f %%' % (
    #         class_names[i], 100 * class_correct[i] / class_total[i]))
    #     #f.write('False Positive of %5s : %.1f %% \n' % (class_names[i], 100 * class_incorrect[i] / class_total[i]))
    #     print('False Positive of %5s : %.1f %%' % (
    #         class_names[i], 100 * class_incorrect[i] / class_total[i]))


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

def binary_print_roc_curve_two(y_test, y_score, fname, n_classes, y_test2, y_score2, fname2, n_classes2, figsize = (8, 6)):

    lw = 2
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        print('------------------------------------')
        f.write('check class{} AUC score for model_ft: {} \n'.format(i, roc_auc_score(y_test[:, i], y_score[:, i])))
        print('Check class{} AUC score: {}'.format(i, roc_auc_score(y_test[:, i], y_score[:, i])))
        print(roc_auc[i])
        if i == 1:
            print(Find_Optimal_Cutoff(y_test[:, i], y_score[:, i], fname))
            # delong_auc, delong_cov = Delong_auc.delong_roc_variance(y_test[:, i], y_score[:, i])
            # print('Delong AUC Score: {}, Delong AUC COV: {}'.format(delong_auc, delong_cov))
            # #print('y score', y_score[:, i])
            # #print('y score2', y_score2[:, i])
            # #print('y score', y_test[:, i])
            # #print('y score2', y_test2[:, i])
            # alpha = .95
            # auc_std = np.sqrt(delong_cov)
            # lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
            # ci = stats.norm.ppf(
            #     lower_upper_q,
            #     loc=delong_auc,
            #     scale=auc_std)
            # ci[ci > 1] = 1
            # print('Delong 95% AUC CI: {}'.format(ci))
            # print(Delong_auc.delong_roc_test(y_test[:, i], y_score[:, i], y_score2[:, i]))


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

#     fpr["macro"] = all_fpr
#     tpr["macro"] = mean_tpr
#     roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    fig = plt.figure(figsize=figsize)

#     plt.plot(fpr["micro"], tpr["micro"],
#             label='micro-average ROC curve (area = {0:0.2f})'
#                     ''.format(roc_auc["micro"]),
#             color='deeppink', linestyle=':', linewidth=4)


#     plt.plot(fpr["macro"], tpr["macro"],
#             label='macro-average ROC curve (area = {0:0.2f})'
#                 ''.format(roc_auc["macro"]),
#             color='navy', linestyle=':', linewidth=4)



    colors = cycle(['aqua','black','cornflowerblue', 'red', 'grey'])
    for i, color in zip(range(2), colors):
        if i == 1:
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))

    fpr2 = dict()
    tpr2 = dict()
    roc_auc2 = dict()
    for i in range(n_classes2):
        fpr2[i], tpr2[i], _ = roc_curve(y_test2[:, i], y_score2[:, i])
        roc_auc2[i] = auc(fpr2[i], tpr2[i])
        print('------------------------------------')
        f.write('check class{} AUC score for model_ft2: {} \n'.format(i, roc_auc_score(y_test2[:, i], y_score2[:, i])))
        print('Check class{} AUC score: {}'.format(i, roc_auc_score(y_test2[:, i], y_score2[:, i])))
        print(roc_auc2[i])
        if i == 0:
            print(Find_Optimal_Cutoff(y_test2[:, i], y_score2[:, i], fname2))
            delong_auc2, delong_cov2 = delong_auc.delong_roc_variance(y_test[:, i], y_score[:, i])
            f.write('Delong AUC Score: {}, Delong AUC COV: {} \n'.format(delong_auc2, delong_cov2))
            print('Delong AUC Score: {}, Delong AUC COV: {}'.format(delong_auc2, delong_cov2))
            #print('y score', y_score[:, i])
            #print('y score2', y_score2[:, i])
            alpha = .95
            auc_std = np.sqrt(delong_cov2)
            lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
            ci = stats.norm.ppf(
                lower_upper_q,
                loc=delong_auc2,
                scale=auc_std)
            ci[ci > 1] = 1
            f.write('Delong 95% AUC CI: {} \n'.format(ci))
            print('Delong 95% AUC CI: {}'.format(ci))
            delong_proc = str(delong_auc.delong_roc_test(y_test[:, i], y_score[:, i], y_score2[:, i]))
            f.write('Delong pROC: {} \n'.format(delong_proc))
            print(delong_auc.delong_roc_test(y_test[:, i], y_score[:, i], y_score2[:, i]))

    # Compute micro-average ROC curve and ROC area
    fpr2["micro"], tpr2["micro"], _ = roc_curve(y_test2.ravel(), y_score2.ravel())
    roc_auc2["micro"] = auc(fpr2["micro"], tpr2["micro"])

    # First aggregate all false positive rates
    all_fpr2 = np.unique(np.concatenate([fpr2[i] for i in range(n_classes2)]))

    # Then interpolate all ROC curves at this points
    mean_tpr2 = np.zeros_like(all_fpr2)


    for i in range(n_classes2):
        mean_tpr2 += interp(all_fpr2, fpr2[i], tpr2[i])

    # Finally average it and compute AUC
    mean_tpr2 /= n_classes2

#     fpr["macro"] = all_fpr
#     tpr["macro"] = mean_tpr
#     roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    fig = plt.figure(figsize=figsize)

#     plt.plot(fpr["micro"], tpr["micro"],
#             label='micro-average ROC curve (area = {0:0.2f})'
#                     ''.format(roc_auc["micro"]),
#             color='deeppink', linestyle=':', linewidth=4)


#     plt.plot(fpr["macro"], tpr["macro"],
#             label='macro-average ROC curve (area = {0:0.2f})'
#                 ''.format(roc_auc["macro"]),
#             color='navy', linestyle=':', linewidth=4)



    colors = cycle(['aqua','black','cornflowerblue', 'red', 'grey'])
    for i, color in zip(range(2), colors):
        if i == 1:
            plt.plot(fpr[i], tpr[i], color=color, lw=lw, linestyle='dotted',
                     label='DenseNet-161' )

            plt.plot(fpr2[i], tpr2[i], color=color, lw=lw,
                     label='Inception-ResNet-v2')




    #plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    ax = plt.axes()

    ax.set_facecolor('white')

    ax.spines['top'].set_color('k')
    ax.spines['top'].set_linewidth(1)
    ax.spines['bottom'].set_color('k')
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_color('k')
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_color('k')
    ax.spines['right'].set_linewidth(1)


    plt.xlim([-0.1, 1.0])
    plt.ylim([0.0, 1.05])

    #plt.title('Some extension of Receiver operating characteristic to binary-class')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.legend(loc="lower right", facecolor = 'white')

    plt.scatter(0.167,0.928, marker='o', color='blue')
    plt.scatter(0.041,0.651, marker='o', color='blue')
    plt.scatter(0.069,0.470, marker='o', color='blue')

    plt.scatter(0.314,0.976, marker='o', color='black')
    plt.scatter(0.013,0.614, marker='o', color='black')
    plt.scatter(0.063,0.952, marker='o', color='black')

    plt.scatter(0.066,0.675, marker='o', color='red')
    plt.scatter(0.145,0.903, marker='o', color='red')
    plt.scatter(0.164,0.771, marker='o', color='red')


    plt.xticks(np.arange(0, 1.2, step=0.2),["{:.1f}".format(x) for x in np.arange(0, 1.2, step=0.2)],size=10)
    plt.yticks(np.arange(0, 1.2, step=0.2),["{:.1f}".format(y) for y in np.arange(0, 1.2, step=0.2)],size=10)
    plt.xlabel('False Positive Rate',size=10)
    plt.ylabel('True Positive Rate',size=10)
    plt.legend(loc="lower right", prop={'size':10},facecolor = 'white')

    #plt.grid(False)


    return fig

def binary_print_roc_curve(y_test, y_score, fname, n_classes, figsize = (8, 6)):

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
        print('Check class{} AUC score: {}'.format(i, roc_auc_score(y_test[:, i], y_score[:, i])))
        print(roc_auc[i])
        if i == 1:
            print(Find_Optimal_Cutoff(y_test[:, i], y_score[:, i], fname))



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

#     fpr["macro"] = all_fpr
#     tpr["macro"] = mean_tpr
#     roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    fig = plt.figure(figsize=figsize)

#     plt.plot(fpr["micro"], tpr["micro"],
#             label='micro-average ROC curve (area = {0:0.2f})'
#                     ''.format(roc_auc["micro"]),
#             color='deeppink', linestyle=':', linewidth=4)


#     plt.plot(fpr["macro"], tpr["macro"],
#             label='macro-average ROC curve (area = {0:0.2f})'
#                 ''.format(roc_auc["macro"]),
#             color='navy', linestyle=':', linewidth=4)



    colors = cycle(['aqua','black','cornflowerblue', 'red', 'grey', 'yellow'])
    #font_prop = font_manager.FontProperties(size=10)
    sn.set_context(font_scale=1.0)
    for i, color in zip(range(2), colors):
        if i == 1:
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))






    #plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    ax = plt.axes()

    ax.set_facecolor('white')

    ax.spines['top'].set_color('k')
    ax.spines['top'].set_linewidth(1)
    ax.spines['bottom'].set_color('k')
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_color('k')
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_color('k')
    ax.spines['right'].set_linewidth(1)


    plt.xlim([-0.1, 1.0])
    plt.ylim([0.0, 1.05])

    plt.title('Some extension of Receiver operating characteristic to binary-class',size=10)
    plt.xticks(np.arange(0, 1.2, step=0.2),["{:.1f}".format(x) for x in np.arange(0, 1.2, step=0.2)],size=10)
    plt.yticks(np.arange(0, 1.2, step=0.2),["{:.1f}".format(y) for y in np.arange(0, 1.2, step=0.2)],size=10)
    plt.xlabel('False Positive Rate',size=10)
    plt.ylabel('True Positive Rate',size=10)
    plt.legend(loc="lower right", prop={'size':10},facecolor = 'white')

    #plt.grid(False)


    return fig

def binary_roc_curve_main(model_ft):
    # obtain outputs of the model
    #alloc_label = True
#     test_dataset = EarDataset(binary_dir=args.data_dir,
#                                    alloc_label = alloc_label,
#                                     transforms=transforms.Compose([Rescale((256, 256)), ToTensor(), Normalize()]))
#     test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    test_loader = dataloaders_test['test']
    y_pred, y_true, y_score, fname = get_output(model_ft, test_loader)
    print(y_pred.shape, y_true.shape, y_score.shape)


    labels = ['Food', 'Nonfood']

    # save the roc curve

    y_onehot = np.zeros((y_true.shape[0], len(labels)), dtype=np.uint8)
    y_onehot[np.arange(y_true.shape[0]), y_true] = 1
    sums = y_onehot.sum(axis=0)
    useless_cols = []
    for i, c in enumerate(sums):
        if c == 0:
            print('useless column {}'.format(i))
            useless_cols.append(i)

    print('Label number check',len(labels))

    useful_cols = np.array([i for i in range(len(labels)) if i not in useless_cols])

    #y_onehot = y_onehot[:,useful_cols]
    #y_score = y_score[:,useful_cols]
    #print(y_onehot)

    fig = binary_print_roc_curve(y_onehot, y_score, fname, useful_cols.shape[0], figsize=(8,6))
    global dirnamefordraw
    dirnamefordraw_roc = 'binary_' + dirnamefordraw
    fig.savefig(os.path.join('models', dirnamefordraw_roc))


def binary_roc_curve_main_two(model_ft, model_ft2):
    # obtain outputs of the model
    #alloc_label = True
#     test_dataset = EarDataset(binary_dir=args.data_dir,
#                                    alloc_label = alloc_label,
#                                     transforms=transforms.Compose([Rescale((256, 256)), ToTensor(), Normalize()]))
#     test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    test_loader = dataloaders_test['test']
    y_pred, y_true, y_score, fname = get_output(model_ft, test_loader)
    y_pred2, y_true2, y_score2, fname2 = get_output(model_ft2, test_loader)
    print(y_pred.shape, y_true.shape, y_score.shape)


    labels = ['Food','Nonfood']

    # save the roc curve

    y_onehot = np.zeros((y_true.shape[0], len(labels)), dtype=np.uint8)
    y_onehot[np.arange(y_true.shape[0]), y_true] = 1
    sums = y_onehot.sum(axis=0)
    useless_cols = []
    for i, c in enumerate(sums):
        if c == 0:
            print('useless column {}'.format(i))
            useless_cols.append(i)

    print('Label number check',len(labels))

    useful_cols = np.array([i for i in range(len(labels)) if i not in useless_cols])

    #y_onehot = y_onehot[:,useful_cols]
    #y_score = y_score[:,useful_cols]
    #print(y_onehot)

    y_onehot2 = np.zeros((y_true2.shape[0], len(labels)), dtype=np.uint8)
    y_onehot2[np.arange(y_true2.shape[0]), y_true2] = 1
    sums2 = y_onehot2.sum(axis=0)
    useless_cols2 = []
    for i, c in enumerate(sums2):
        if c == 0:
            print('useless column {}'.format(i))
            useless_cols2.append(i)

    print('Label number check',len(labels))

    useful_cols2 = np.array([i for i in range(len(labels)) if i not in useless_cols2])
    global dirnamefordraw
    dirnamefordraw_roc_two = 'binary_ROC_curve_two_' + dirnamefordraw
    fig = binary_print_roc_curve_two(y_onehot, y_score, fname, useful_cols.shape[0], y_onehot2, y_score2, fname2, useful_cols2.shape[0], figsize=(8,6))

    fig.savefig(os.path.join('models', dirnamefordraw_roc_two), dpi=200, format='png')

opt_conf = []
targetList = []
#from cf_matrix import make_confusion_matrix
#from sklearn.metrics import plot_confusion_matrix

def Find_Optimal_Cutoff(target, predicted, fname):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------
    list type, with optimal cutoff value

    """
    class_names = ['Food', 'Nonfood']
    confusion_matrix = cm.ConfusionMeter(len(class_names))
    o1 = open("Opt_TruePositve.csv", "w")
    o2 = open("Opt_FalsePositive.csv", "w")
    o3 = open("Opt_FalseNegative.csv", "w")
    o4 = open("Opt_TrueNegative.csv", "w")

    #print(fname)
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i),
                       'tpr' : pd.Series(tpr), 'fpr' : pd.Series(fpr) })
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    print(roc_t)


    y_score_opt = np.where(predicted > list(roc_t['threshold'])[0], 1, 0)

    from sklearn.metrics import confusion_matrix
    categories = ['nonspecific', 'hyperplastic', 'ssa', 'adenoma', 'tsa', 'carcinoma']
    opt_conf = confusion_matrix(target, y_score_opt, labels=np.unique(target))

    plt.subplots(figsize=(fig_horizontal,fig_vertical))

    con_m = opt_conf

    df_con_m = pd.DataFrame(con_m, index= [i for i in class_names], columns = [i for i in class_names])
    ### each labels percentage
    cm1 = con_m
    cm_sum = np.sum(cm1, axis=1, keepdims=True)
    cm_perc = cm1 / cm_sum.astype(float) * 100
    annot = np.empty_like(opt_conf).astype(str)
    nrows, ncols = cm1.shape

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
    ### each labels percentage

    group_counts = ["{0:0.0f}".format(value) for value in opt_conf.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in opt_conf.flatten()/np.sum(opt_conf)]

    labels = [f"{v1} ({v2})" for v1, v2 in zip(group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(2,2)

    categories = ['nonspecific', 'hyperplastic', 'ssa', 'adenoma', 'tsa', 'carcinoma']
    #make_confusion_matrix(opt_conf, categories=categories, cmap='Blues')

    # sn.set(font_scale= 2.5)
    sn.heatmap((cm1.T / cm1.sum(axis=1)).T, annot=annot, fmt='', annot_kws={"size" : 60}, square=True, cmap='Blues',cbar = True, vmin=0, vmax=1)
    #sn.heatmap(df_con_m, annot=True,fmt='g', annot_kws={"size" : 40}, cbar = False, cmap="Blues")
    #sn.heatmap(df_con_m/np.sum(df_con_m), annot=True, fmt='.2%', cbar = False, cmap='Blues')
    #sn.heatmap(inte_df_con_m, annot=True, fmt='.2%', cbar = False, cmap='Blues')
    plt.yticks(rotation = 0)
    #plt.xticks(rotation = 45)
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    global dirnamefordraw
    dirnamefordraw_opt_conf = 'binary_opt_confusion_' + dirnamefordraw + '.png'
    plt.savefig(os.path.join('models', dirnamefordraw_opt_conf),dpi=100, format='png')
    #fig.savefig(os.path.join('models', 'mucosal_resnet_exp2'))

    ##################### 결과 파일명 뽑기
    sourceNDArray = np.array(y_score_opt)
    optList = sourceNDArray.tolist()

    targetNDArray = np.array(target)
    tarList = targetNDArray.tolist()




    for i in range(len(optList)):
        filename = os.path.basename(fname[i])
        pathname = os.path.dirname(fname[i])
        patientname = filename.split(' ')[0]

        if tarList[i] == 0 and tarList[i] == optList[i]:
            o1.write("{}, {}, {}, label: {}, predicted: {}, 1\n".format(pathname, filename, patientname, tarList[i], optList[i]))
        elif tarList[i] == 0 and tarList[i] != optList[i]:
            o2.write("{}, {}, {}, label: {}, predicted: {}, 2\n".format(pathname, filename, patientname, tarList[i], optList[i]))
        elif tarList[i] == 1 and tarList[i] == optList[i]:
            o4.write("{}, {}, {}, label: {}, predicted: {}, 4\n".format(pathname, filename, patientname, tarList[i], optList[i]))
        elif tarList[i] == 1 and tarList[i] != optList[i]:
            o3.write("{}, {}, {}, label: {}, predicted: {}, 3\n".format(pathname, filename, patientname, tarList[i], optList[i]))

#         if tarList[i] == 1 and tarList[i] != optList[i]:
#             o1.write("{}, {}, label: {}, predicted: {}, 1\n".format(filename, patientname, tarList[i], optList[i]))
# #         elif tarList[i] == 0 and tarList[i] != optList[i]:
# #             o2.write("{}, {}, label: {}, predicted: {}, 2\n".format(filename, patientname, tarList[i], optList[i]))
# #         elif tarList[i] == 1 and tarList[i] == optList[i]:
# #             o4.write("{}, {}, label: {}, predicted: {}, 4\n".format(filename, patientname, tarList[i], optList[i]))
# #         elif tarList[i] == 1 and labels != optList[i]:
# #             o3.write("{}, {}, label: {}, predicted: {}, 3\n".format(filename, patientname, tarList[i], optList[i]))

    o1.close()
    o2.close()
    o3.close()
    o4.close()


    # print('optimal accuracy : \n', accuracy_score(target, y_score_opt))
    # print('optimal conf mat : \n', confusion_matrix(target, y_score_opt))
    # tn, fp, fn, tp = confusion_matrix(target, y_score_opt).ravel() # sensitivty / specificity / ppv / npv
    # # print(tn, fp, fn, tp)
    # print('sensitivity : ', tp / (tp + fn))
    # print('specificity : ', tn / (fp + tn))
    # print('PPV : ', tp / (tp + fp))
    # print('NPV : ', tn / (tn + fn))
    # print('------------------------------------')

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


def getter_confusion_matrix(model_ft):
    #Get the confusion matrix for testing data
    confusion_matrix = cm.ConfusionMeter(len(class_names))
    roc_score = 0
    data = []
    number = 0
    a = open("TruePositive.csv", "w")
    b = open("FalsePositive.csv", "w")
    c = open("FalseNegative.csv", "w")
    d = open("TrueNegative.csv", "w")


    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders_test['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model_ft(inputs)

            _, predicted = torch.max(outputs, 1)

            #print(predicted)
            #print(predicted.item())
            #print(dataloaders_val['test'].dataset.samples[i])

            sample_fname, p = dataloaders_test['test'].dataset.samples[i]

            filename = os.path.basename(sample_fname)
            patientname = filename.split(' ')[0]
            #print(filename)
            #print(filename.split(' ')[0])

            

            if labels == 0 and labels == predicted.item():
                a.write("{}, {}, label: {}, predicted: {}, 1\n".format(filename, patientname, labels.item(), predicted.item()))
            elif labels == 0 and labels != predicted.item():
                b.write("{}, {}, label: {}, predicted: {}, 2\n".format(filename, patientname, labels.item(), predicted.item()))
            elif labels == 1 and labels == predicted.item():
                d.write("{}, {}, label: {}, predicted: {}, 4\n".format(filename, patientname, labels.item(), predicted.item()))
            elif labels == 1 and labels != predicted.item():
                c.write("{}, {}, label: {}, predicted: {}, 3\n".format(filename, patientname, labels.item(), predicted.item()))




            confusion_matrix.add(predicted, labels)
            #torch.topk(nn.Softmax(dim=1)(model(img_tensor.cuda())), 1)


        print(confusion_matrix.conf)

    a.close()
    b.close()
    c.close()
    d.close()
    return confusion_matrix.conf

def toCopy_prediction(model_ft, diff_list):
    # Copy prediction
    
    # a = open("1.csv", "w")
    # b = open("2.csv", "w")
    # c = open("3.csv", "w")
    # d = open("4.csv", "w")


    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(dataloaders_test['test'])):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model_ft(inputs)

            _, predicted = torch.max(outputs, 1)

            #print(predicted)
            #print(predicted.item())
            #print(dataloaders_val['test'].dataset.samples[i])

            #sample_fname, p = dataloaders_test['test'].dataset.samples[i]
            
            sample_fname, p = dataloaders_test['test'].dataset.samples[i]

            predicted_filename = os.path.basename(sample_fname)
            
            #print(filename)

            dst_folder = '/home/jovyan/data/classified/{}/'.format(predicted.item())
            
            if not os.path.isdir(dst_folder):
                 os.makedirs(dst_folder, exist_ok=True)
            
            count = 0
            
#                 for filename in filenames:
#                     if not filename in whole_path_list:
#                         shutil.copy(sample_fname, dst_folder + predicted_filename)
            
            if predicted_filename in diff_list:
                shutil.copy(sample_fname, dst_folder + predicted_filename)
            count += 1
    
    print("All of copied!!!")
def toCopy_prediction_read_tensor(model_ft, diff_list, whole_path):
    # Copy prediction
    
    # a = open("1.csv", "w")
    # b = open("2.csv", "w")
    # c = open("3.csv", "w")
    # d = open("4.csv", "w")


    with torch.no_grad():
        
        for idx in tqdm(range(len(diff_list))):
            whole_filename = os.path.join(whole_path, diff_list[idx])
            img_tensor = read_tensor(whole_filename)
            
            #_, predicted = torch.max(img_tensor, 1)
            
            pp, cc = torch.topk(nn.Softmax(dim=1)(model_ft(img_tensor.cuda())), 1)
            
            dst_folder = '/home/jovyan/data/classified/{}/'.format(cc)
            
            if not os.path.isdir(dst_folder):
                 os.makedirs(dst_folder, exist_ok=True)
#             whole_filename = os.path.join(whole_path, diff_list[idx])
            shutil.copy(whole_filename, dst_folder + diff_list[idx])
            
    print("All of copied!!!")

# def draw_heatmap(con_mat):
#     #Confusion matrix as a heatmap
#     con_m = con_mat
#     df_con_m = pd.DataFrame(con_m, index= [i for i in class_names], columns = [i for i in class_names])

#     cm1 = con_m

#     cm_sum = np.sum(cm1, axis=1, keepdims=True)
#     cm_perc = cm1 / cm_sum.astype(float) * 100
#     annot = np.empty_like(cm1).astype(str)
#     nrows, ncols = cm1.shape

#     for i in range(nrows):
#         for j in range(ncols):
#             c = cm1[i, j]
#             p = cm_perc[i, j]
#             if i == j:
#                 s = cm_sum[i]
#                 annot[i, j] = '%d (%d%%)' % (c, p)
#             elif c == 0:
#                 annot[i, j] = ''
#             else:
#                 annot[i, j] = '%d (%d%%)' % (c, p)

#     cm1 = pd.DataFrame(cm1, index= [i for i in class_names], columns = [i for i in class_names])

#     sn.set(font_scale= 1.1)
#     #sn.heatmap(con_m, annot=True,fmt='g', annot_kws={"size" : 10}, cbar = True, cmap="Blues", vmin=0, vmax=1)
#     sn.heatmap((cm1.T / cm1.sum(axis=1)).T, annot=annot, fmt='', annot_kws={"size" : 10}, cmap='Blues',cbar = False, vmin=0, vmax=1)
#     #sn.heatmap(df_con_m, vmin=0, vmax=1)

def draw_heatmap(con_mat):
    #Confusion matrix as a heatmap
    #con_m = con_mat
    class_names = image_datasets['test'].classes
    #df_con_m = pd.DataFrame(con_m, index= [i for i in class_names], columns = [i for i in class_names])

    cm1 = con_mat

    #print(cm1)

    cm_sum = np.sum(cm1, axis=1, keepdims=True)
    cm_perc = cm1 / cm_sum.astype(float) * 100
    annot = np.empty_like(cm1).astype(str)
    nrows, ncols = cm1.shape
    class_names = ['Food', 'NonFood']

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

    plt.subplots(figsize=(fig_horizontal,fig_vertical))
    #sn.set(font_scale= 1.5)
    #sn.heatmap(df_con_m, annot=True,fmt='g' ,  annot_kws={"size" : 10}, cbar = False, cmap="Blues")
    sn.heatmap((cm1.T / cm1.sum(axis=1)).T, annot=annot, fmt='', annot_kws={"size" : 60}, cmap='Blues', square=True, cbar=True, vmin=0, vmax=1)
    plt.yticks(rotation = 0)
    #plt.xticks(rotation = 45)
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    global dirnamefordraw
    dirnamefordraw_conf = 'binary_confusion_' + dirnamefordraw + '.png'
    plt.savefig(os.path.join('models', dirnamefordraw_conf),dpi=100, format='png')


if __name__ == "__main__":

    #modelnumber = 3
    sn.set(font_scale= 4.5)
    #overall_accuracy(modelnumber)
    
    model_ft = torch.load('./DenseNet_best_defaultsize.pth')
    print("Copying Starts!!")
    
    classified_list = []
    filename_list = []
    classified_path = "/home/jovyan/data/classified"
    whole_path = "/home/jovyan/data/food-dataset/test/food"
    whole_path_list = os.listdir(whole_path)
    
    for idx in range(len(whole_path_list)):
        #print(whole_name)
        whole_path_list.append(os.path.basename(whole_path_list[idx]))
    
    for dirpath, dirnames, filenames in os.walk(classified_path):
        for filename in filenames:
            classified_list.append(os.path.join(dirpath, filename))
            filename_list.append(filename)
    
    
    diff_list = np.setdiff1d(whole_path_list, filename_list)
    
    
    
    toCopy_prediction_read_tensor(model_ft, diff_list, whole_path)
    
    ## multiprocessing
#     num_processes =4
#     mp.set_start_method('spawn', force=True)
#     model_ft.share_memory()
#     processes = []
#     for rank in range(num_processes):
#         p = mp.Process(target=toCopy_prediction, args=(model_ft,))
#         p.start()
#         processes.append(p)
#     for p in processes:
#         p.join()
    
    #class_wise_tesing_accuracy()
    #binary_roc_curve_main()
    #binary_roc_curve_main_two()

    #draw_heatmap(getter_confusion_matrix())
