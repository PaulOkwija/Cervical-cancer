from keras import backend as K


def dice_coef(y_true, y_pred):
    smooth = 1e-5
    y_true_f = float(K.flatten(y_true))
    y_pred_f = float(K.flatten(y_pred))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth)/(K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true,y_pred):
    smooth = 1e-5
    return (1-dice_coef(y_true,y_pred))

def IoU(y_true,y_pred):
    smooth = 1e-5
    y_true_f = float(K.flatten(y_true))
    y_pred_f = float(K.flatten(y_pred))
    intersection = K.sum(y_true_f*y_pred_f)
    result = (intersection + smooth)/(K.sum(y_true_f)+K.sum(y_pred_f)-intersection + smooth)
    return result

    
def IoU_loss(y_true,y_pred):
    smooth = 1e-5
    return (1 - IoU(y_true,y_pred))

