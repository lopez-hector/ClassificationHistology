# Classification of Colorectal Histology Images
## Classification of 7 tissue types, including tumors
___

This repo contains script to classify histology images from the following tissues:
tumor, lympho, mucosa, stroma, complex, and adipose.

# Models
1) Small network built from residual convolutional and identity blocks.

# Data

Trained on kaggle dataset [here](https://www.kaggle.com/datasets/kmader/colorectal-histology-mnist).
  
Citation:  
Kather JN, Weis CA, Bianconi F, Melchers SM, Schad LR, Gaiser T, Marx A, Zollner F: Multi-class texture analysis in colorectal cancer histology (2016), Scientific Reports (in press)

Example images:  
![](Images/example_tissues.png)
# Performance
## Custom ResNet50 Inspired
### Precision and Recall
![img.png](Images/img.png)  
where inVal is number of images in validation set, noPredicted is TP+FP, and noPredictedCorrectly is TP.
### Example of Misclassified Images
![](Images/misclassified_tissues.png)
### Confusion Matrix
![img_1.png](Images/img_1.png)
### Analysis
The model performs decently well across the various classes. The addition of the complex 
tissues, results in the majority of misclassification events (precision of complex tissue < 79%). Most importantly, 
the complex tissue class accounts for all 7/114 misclassified tumors. If this model 
was to be considered in practice, it would be critical to improve its recall
of tumor tissues or, more simply, double check complex tissue classifications for possible
tumors.  
Here is a comparison on complex and tumor tissues from the dataset.  
![](Images/ComplexTissues.png)