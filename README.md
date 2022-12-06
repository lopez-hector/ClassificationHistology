# Classification of Tissue Types from Colorectal Histology Images
## Classification of 7 tissue types, including tumors
Identifying tissue types from histological images is a critical step in pathology. Computer vision techniques enable rapid and automatic classification of tissue types. This repo contains details on CV model(s) and notebooks for the inference and classification of histology images from the following tissues: tumor epithelium (tumor), immune cells (lympho), mucosal glands (mucosa), simple stroma (stroma), complex stroma (complex), and adipose.

notebooks with exploratory data preparation, model training, and analysis:  
CNN: [here](https://www.kaggle.com/code/hectorlopezhernandez/colorectalhistologymodel) and [here](https://www.kaggle.com/code/hectorlopezhernandez/analysis-colorectaldata)  
Transformers: [here](https://colab.research.google.com/drive/1bQk-LHVE9YFDPCjuYHJF-zSOf1aw38hq?usp=sharing)

# Models
1) CNN built from residual convolutional and identity blocks.
  - fully trained on labeled histology images alone
2) ViT: Vision transformer encoder model (from [huggingface](google/vit-base-patch16-224))
  - pretrained by google with supervised methods on ImageNet data
  - transfer learning: trained new classification head on the base model
3) BEiT: Vision transfomer encoder model (from [huggingface](microsoft/beit-base-patch16-224-pt22k-ft22k)) pretrained using self-supervised methods.
  - pretrained by microsoft with self-supervised sample-contrastive methods on ImageNet data. (learns image representations without labeled data)
  - transfer learning: trained new classification head on the base model

# Data

Trained on kaggle dataset [here](https://www.kaggle.com/datasets/kmader/colorectal-histology-mnist).
  
Citation:  
Kather JN, Weis CA, Bianconi F, Melchers SM, Schad LR, Gaiser T, Marx A, Zollner F: Multi-class texture analysis in colorectal cancer histology (2016), Scientific Reports (in press)

Example images for tissues and controls:  

![](Images/example_tissues.png)

# Overall Comparison
ViT was the best performing model with only 4 epochs of supervised fine-tuning on 4,000 labeled histology images. BEiT came in close second but required 8 epochs of fine-tuning to achieve a competitive performance. 

![Untitled](https://user-images.githubusercontent.com/65481379/206043074-614cf3e6-92df-4eb1-b6f7-ad6208a9c100.png)


## ROC Curve Comparison
While all models did well with > 90% accuracy, the transformer models performed slighlty better as classifiers. The recall of tumor tissues is the most important metric in this dataset. ViT achieved  96% recall for the tumor tissue class.

<img width="1600" alt="ModelComparisionROC" src="https://user-images.githubusercontent.com/65481379/205772531-d6b9ccd2-cf08-4b45-9a5a-adc144f19c76.png">


# CNN Performance
Accuracy: 91.3  
Training set: 4,000 Images  
Validation set: 1,000 Images
## Precision and Recall
![img.png](Images/img.png)  
where inVal is number of images in validation set, noPredicted is TP+FP, and noPredictedCorrectly is TP.

## Confusion Matrix
![img_1.png](Images/img_1.png)
## Analysis
![](Images/ROCcurveOvR.png)

# ViT Performance
Accuracy: 0.945  
Training set: 4,000 Images  
Validation set: 1,000 Images  

## Precision and Recall
![Untitled](https://user-images.githubusercontent.com/65481379/205764572-c355880c-0ec1-4b07-9054-264f3ffac44c.png)
where inVal is number of images in validation set, noPredicted is TP+FP, and noPredictedCorrectly is TP.

## Confusion Matrix
![Untitled 1](https://user-images.githubusercontent.com/65481379/205764655-89913108-920e-406d-83e6-651ee189c015.png)

## Analysis
![Untitled 2](https://user-images.githubusercontent.com/65481379/205764700-728f0858-890c-4baf-87e7-c5182d83bab0.png)

# BEiT Performance
Accuracy: 0.928  
Training set: 4,000 Images  
Validation set: 1,000 Images  

## Precision and Recall
![Untitled 3](https://user-images.githubusercontent.com/65481379/205764985-18336cb4-95d1-4213-822a-9cc8fbaa8756.png)

where inVal is number of images in validation set, noPredicted is TP+FP, and noPredictedCorrectly is TP.

## Confusion Matrix
![ConfusionMatrix](https://user-images.githubusercontent.com/65481379/205765021-351ead58-28b1-4789-b7c5-7c5549965b3e.png)


## Analysis

![ROC_curve_OvR](https://user-images.githubusercontent.com/65481379/205765125-537c7bce-5135-4413-bc80-3c9a6e6137a4.png)


# Overall Recommendations
If this model was to be considered in practice, it would be critical to achieve 100% recall of tumor tissues. I would
recommend both complex and tumor tissues for further analysis, given the presence of some tumor cells in the complex stroma
data [Kather et al.]. This classification strategy would compromise precision to achieve 100% recall with our existing dataset 
on tumors. 

## A comparison on complex and tumor tissues from the dataset 

The models perform well across the various classes, with the ViT model achieving >= 90% recall across all classes. The complex 
tissue class, resulted in the majority of misclassification events (lowest precision). Since the complex tissue
is comprised of stroma, single tumor, and single immune cells, the models likely learn similar feature representations between the tissue classes. This overlap is obvious in the results, where the majority of False positives and false negatives for tumors fell into the complex tissue class.

### Comparison of Tumor Tissues and Complex Tissues

![](Images/ComplexTissues.png)

## Alternative classification based on threshold probability

Multi-class classification is typically based on the argmax of the logits (or softmax) output of the models. Given that the model outputs probabilities of each class, I explored the classification of tumors based on a threshold probability on the tumor class alone (If P(tumor) > threshold -> tumor). This strategy is more aligned with the absolute necessity of identifying tumors, over classification of other tissues. Closer inpection of the ROC curves in the following figure shows the TPR reach 1 with only a moderate increase in FPR. In this applicaion, the increase in FPR would be acceptable to increase confidence in detection of all tumors.


![image](https://user-images.githubusercontent.com/65481379/205770282-7745bbeb-9198-4f53-9ef5-4f21652494fd.png)

# Example of Misclassified Images
![](Images/misclassified_tissues.png)
