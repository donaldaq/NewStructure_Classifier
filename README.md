# NewStructure Classifier for Image Deep Learning

This is image classifier using PyTorch. NewStructure Classifier for Image Deep Learning can use image deep learning. 



#### Features

- Image dataloader is [image folder structure](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder) in torchvision
- image transforms use torchvision
- Early stopping control using only validation loss 
- batch balanced each classes control 
-  Result metrics: 
  - Overall Accuracy
  - Class wise Accuracy
  - AUC Score / ROC Curve graph and saving image file
  - **Optimal threshold** for Maximum Sensitivity and Maximum 1-Specificity and saving confusion matrix
  - Confusion Matrix 
  - Delong Test(pROC) for two learning weight models 

- Learning log: tensorboard, text file(learning log and result log in Result folder)



#### Main information 

- PyTorch version: 1.3 or above(except for [default_classifier_transformer.py](https://github.com/donaldaq/classifier-pytorch/blob/master/default_classifier_transformer.py): 1.7)
- Torchvision: 0.4.2
- Python: 3.6.9
- This repository is separated binary and multiclass test result code



#### Folder Structure 

```buildoutcfg
+-- root
|   +-- train
|       +-- class1
|           +-- img1.jpg
|           +-- img2.jpg
|           +-- img3.jpg
|       +-- class2
|       +-- class3
|   +-- test
|       +-- class1
|       +-- class2
|       +-- class3
|   +-- val
|       +-- class1
|       +-- class2
|       +-- class3
```





#### Reference

- Image transform: https://github.com/lucidrains/vit-pytorch
