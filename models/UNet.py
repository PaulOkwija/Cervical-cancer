#from platform import architecture
#from tkinter import _Padding
import tensorflow as tf
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras import backend as K

from keras.models import Model
from keras.layers import Input,add , Conv2D,Activation, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras import backend as K
#######################################################################################################
'''
We are building the architecture using blocks unlike the previous model which just has stacked layers
First we define these blocks
'''
#Convolution block
def conv_block(s, filter_size, size, dropout, k_init, act, batch_norm=False):

    #1

    conv = Conv2D(size, (filter_size, filter_size) , activation=act, kernel_initializer=k_init, padding='same')(s)
    if batch_norm is True:
        conv = BatchNormalization(axis=3)(conv)
 #Dropout
    if dropout > 0:
        conv = Dropout(dropout)(conv)
    #2

    conv = Conv2D(size, (filter_size, filter_size), activation=act, kernel_initializer=k_init, padding='same')(conv)
    if batch_norm is True:
        conv = BatchNormalization(axis=3)(conv)
    
    return conv



def Unet_baseline_up2d(input_shape,f_num,k_init, act, NUM_CLASSES=1, dropout_rate=0.1, batch_norm=False):
    '''
    ResUNet 
    
    '''
    # network structure
    FILTER_NUM = f_num # number of basic filters for the first layer
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters
    
    inputs = layers.Input(input_shape, dtype=tf.float32)
    #inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv_16 = conv_block(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate,k_init, act,batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 2
    conv_32 = conv_block(pool_16, FILTER_SIZE, 2*FILTER_NUM, dropout_rate,k_init, act, batch_norm)
    pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 3
    conv_64 = conv_block(pool_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate,k_init, act, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 4
    conv_128 = conv_block(pool_64, FILTER_SIZE, 8*FILTER_NUM, dropout_rate,k_init, act, batch_norm)
    pool_128 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 5, convolution only
    conv_256 = conv_block(pool_128, FILTER_SIZE, 16*FILTER_NUM, dropout_rate,k_init, act, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    up_256 = Conv2DTranspose(8*FILTER_NUM, (2, 2), strides=(2, 2), padding='same')(conv_256)
    up_256 = concatenate([up_256, conv_128])
    up_conv_128 = conv_block(up_256, FILTER_SIZE, 8*FILTER_NUM, dropout_rate,k_init, act, batch_norm)
    # UpRes 7
    up_128 = Conv2DTranspose(4*FILTER_NUM, (2, 2), strides=(2, 2), padding='same')(up_conv_128)
    up_128 = concatenate([up_128, conv_64])
    up_conv_64 = conv_block(up_128, FILTER_SIZE, 4*FILTER_NUM, dropout_rate,k_init, act, batch_norm)
    # UpRes 8
    up_64 = Conv2DTranspose(2*FILTER_NUM, (2, 2), strides=(2, 2), padding='same')(up_conv_64)
    up_64 = concatenate([up_64, conv_32])
    up_conv_32 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate,k_init, act, batch_norm)
    # UpRes 9
    up_32 = Conv2DTranspose(FILTER_NUM, (2, 2), strides=(2, 2), padding='same')(up_conv_32)
    up_32 = concatenate([up_32, conv_16])
    up_conv_16 = conv_block(up_32, FILTER_SIZE, FILTER_NUM, dropout_rate,k_init, act, batch_norm)

    # 1*1 convolutional layers
    conv_final = Conv2D(NUM_CLASSES, kernel_size=(1,1), activation='sigmoid')(up_conv_16)
    #conv_final = layers.BatchNormalization(axis=3)(conv_final)
    #Change to softmax for multichannel

    # Model integration
    model = Model(inputs, conv_final, name="Unet_up_2")
    # model.compile(optimizer='adam', loss=[jaccard_coef_loss], metrics=[jaccard_coef])
    # model.summary()
    return model