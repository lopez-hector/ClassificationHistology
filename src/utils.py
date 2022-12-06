import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import seaborn as sns

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import ConfusionMatrixDisplay

import tensorflow as tf

import os

def print_performance(val_label, predictions, id2label, dir_where):
    # print dataframe with relevant performance metrics for model
    # val_label(ndarray): validation set labels
    # predictions(ndarray): model predictions (id)
    # id2label (dict) - maps ids to labels
    # dir_where: save directory
    
    df = pd.DataFrame()
    df.index = id2label.values()
    
    total_in_ds = []
    total_predicted = []
    predicted_correctly = []

#     print(val_label==0)
    
    for i, class_ in enumerate(id2label):
        total_in_ds.append(sum(val_label==i))
        total_predicted.append(sum(predictions == i)) #predictions made for a given class
        predicted_correctly.append(sum(val_label[val_label==i] == predictions[val_label==i]))
        
    
    recall = recall_score(val_label, predictions,average=None)
    precision = precision_score(val_label, predictions, average=None)
    
    df['inVal'] = total_in_ds
    df['noPredicted'] = total_predicted
    df['noPredictedCorrectly'] = predicted_correctly
    df['recall'] = recall
    df['precision'] = precision
    display(df)
    df.to_csv(os.path.join(dir_where,'model_metrics.csv'))

def save_predictions(logits, preds, ground_truths, file_paths, dir_where):
  ### used to save dictionary of model predictions, labels, logits, and file_paths to images
  
  # logits (ndarray)
  # preds (ndarray): model predictions 
  # ground_truths (ndarray)
  # file_paths (list of strings)
  # dir_where: save directory
  
  data = {
          'logits':logits.tolist(),
          'preds':preds.tolist(),
          'ground_truths':ground_truths.tolist(),
          'file_paths': file_paths
          }

  with open(os.path.join(dir_where, 'predictions_data.json'), 'w') as f:
    json.dump(data, f, indent=2)



def print_confusion(val_label, predictions, id2label, dir_where):
  # val_label(ndarray): validation set labels
  # predictions(ndarray): model predictions (id)
  # id2label (dict) - maps ids to labels
  # dir_where: save directory
  
  # save confusion matrix to disk
  CM = tf.math.confusion_matrix(
      val_label,
      predictions,
      num_classes=None,
      weights=None,
      dtype=tf.dtypes.int32,
      name=None
  )
  with open(os.path.join(dir_where, 'confusion_matrix.json'), 'w') as f:
    json.dump(CM.numpy().tolist(), f, indent=2)
  
  # print confusion matrix and save image
  plt.figure(figsize=(15, 15))
  ax = plt.axes()
  ConfusionMatrixDisplay.from_predictions(val_label, predictions,display_labels=id2label.values(), ax=ax)
  plt.savefig(os.path.join(dir_where,'ConfusionMatrix.png'))


def roc_curves(val_label, logits, id2label, dir_where):
  # logits (list)
  # val_label(ndarray): validation set labels
  # id2label (dict) - maps ids to labels
  # dir_where: save directory
  
  # Binarize predictions
  binary_ground_truth = label_binarize(val_label, classes=[i for i in range(8)]) # returns (m, n_classes)
  
  # Show scores for each class
  logits = np.array(logits)

  # Get roc curve for each column (class)
  fpr, tpr, thresholds, roc_auc = {}, {}, {}, []

  for i in range(len(id2label.values())):
      
      fpr[i], tpr[i], thresholds[i] = roc_curve(binary_ground_truth[:, i], logits[:, i], drop_intermediate=False) # Returns ndarray
      roc_auc.append(auc(fpr[i], tpr[i])) # Returns Float
  
  # Plot Each ROC Curve on same figure
  plt.figure(figsize=(8,8))
  ax = plt.axes()
  np.set_printoptions(precision=2)

  for i in range(len(id2label.values())):
      sns.lineplot(x=fpr[i], y=tpr[i], label=f"ROC curve of class {id2label[i]} (area = {roc_auc[i]:0.2f})")
      
  plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
  plt.xlim([0.0, 1])
  plt.ylim([0.0, 1.05])
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title("ROC")
  plt.legend(loc="lower right")
  plt.savefig(os.path.join(dir_where,'ROC_curve_OvR.png'))


def get_histology_images(save_dir):
  '''
    download and unzip PNG histology data to save_dir
    MAKE SURE YOUR API KEY IS IN SAVE_DIR
    
    save_dir: save path

  '''
  # get_histology_images("/content/drive/MyDrive/ColorectalHistology/")
  os.chdir(save_dir)

  os.environ['KAGGLE_CONFIG_DIR'] = save_dir

  # Dataset converted from here https://www.kaggle.com/datasets/kmader/colorectal-histology-mnist
  os.system('kaggle datasets download -d hectorlopezhernandez/colorectal-histology-pngs')
  os.system('unzip colorectal-histology-pngs.zip')
