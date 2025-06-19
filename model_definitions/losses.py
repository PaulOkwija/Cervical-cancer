"""
losses.py
-----------
Custom loss functions and metrics for image segmentation tasks, especially for medical image segmentation (e.g., cervical cancer lesion segmentation).

Functions:
- dsc: Dice Similarity Coefficient (DSC) metric.
- dice_loss: Dice loss (1 - DSC).
- bce_dice_loss: Combined Binary Crossentropy and Dice loss.
- confusion: Computes precision and recall from predictions and ground truth.
- tp: True positive rate.
- tn: True negative rate.
- tversky: Tversky index for imbalanced segmentation.
- tversky_loss: Tversky loss (1 - Tversky index).
- focal_tversky: Focal Tversky loss for highly imbalanced data.

All functions are compatible with Keras/TensorFlow backend and can be used as custom losses or metrics in model training.
"""
from keras.losses import binary_crossentropy
import keras.backend as K
import tensorflow as tf 

# Small constants for numerical stability
epsilon = 1e-5
smooth = 1

def dsc(y_true, y_pred):
    """
    Dice Similarity Coefficient (DSC).
    Args:
        y_true: Ground truth mask.
        y_pred: Predicted mask.
    Returns:
        Dice coefficient score (float).
    """
    smooth = 1.
    y_true_f = float(K.flatten(y_true))
    y_pred_f = float(K.flatten(y_pred))
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    """
    Dice loss (1 - DSC).
    """
    loss = 1 - dsc(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    """
    Combined Binary Crossentropy and Dice loss.
    """
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

def confusion(y_true, y_pred):
    """
    Computes precision and recall from predictions and ground truth.
    Returns:
        prec: Precision
        recall: Recall
    """
    smooth=1
    y_pred_pos = K.clip(y_pred, 0, 1)
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.clip(y_true, 0, 1)
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg) 
    prec = (tp + smooth)/(tp+fp+smooth)
    recall = (tp+smooth)/(tp+fn+smooth)
    return prec, recall

def tp(y_true, y_pred):
    """
    True positive rate.
    """
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pos = K.round(K.clip(y_true, 0, 1))
    tp = (K.sum(y_pos * y_pred_pos) + smooth)/ (K.sum(y_pos) + smooth) 
    return tp 

def tn(y_true, y_pred):
    """
    True negative rate.
    """
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos 
    tn = (K.sum(y_neg * y_pred_neg) + smooth) / (K.sum(y_neg) + smooth )
    return tn 

def tversky(y_true, y_pred):
    """
    Tversky index for imbalanced segmentation.
    Args:
        y_true: Ground truth mask.
        y_pred: Predicted mask.
    Returns:
        Tversky index (float).
    """
    y_true_pos = float(K.flatten(y_true))
    y_pred_pos = float(K.flatten(y_pred))
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    """
    Tversky loss (1 - Tversky index).
    """
    return 1 - tversky(y_true,y_pred)

def focal_tversky(y_true,y_pred):
    """
    Focal Tversky loss for highly imbalanced data.
    """
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)
