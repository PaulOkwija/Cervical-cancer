from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16


def conv_block(drop,inputs, num_filters):
  x = layers.Conv2D(num_filters, 3,activation='relu', kernel_initializer='he_normal',padding="same")(inputs)
  x = layers.BatchNormalization()(x)
  # x = layers.Activation("relu")(x)
  x = layers.Dropout(drop)(x)

  x = layers.Conv2D(num_filters, 3, activation='relu',kernel_initializer='he_normal',padding="same")(x)
  x = layers.BatchNormalization()(x)
  # x = layers.Activation("relu")(x)
  return x


def decoder_block(drop, inputs, skip_features, num_filters):
  x = layers.Conv2DTranspose(num_filters, (2,2), strides=2, kernel_initializer='he_normal',padding="same")(inputs)
  #print("1",x.shape, skip_features.shape)
  x = layers.concatenate([x, skip_features])
  #print("2",x.shape, skip_features.shape)
  x = conv_block(drop,x, num_filters)
  #print("3",x.shape, skip_features.shape)

  return x


def build_vgg16_unet(drop,input_shape):
  inputs = layers.Input(shape=input_shape)
  vgg = VGG16(include_top=False, weights='imagenet', input_tensor=inputs, input_shape = input_shape, pooling = 'max' )
  # vgg = VGG19(include_top=False, weights=None, input_tensor=inputs, input_shape = input_shape, pooling = 'max' )
  # resnet = resnet50(include_top=False, weights=None, input_tensor=inputs, input_shape = input_shape, pooling = 'max' )
  # incep = inceptionv3(include_top=False, weights=None, input_tensor=inputs, input_shape = input_shape, pooling = 'max' )
  vgg.trainable = False
  # print(vgg.summary())

 
  #vgg = build_vgg16_unet(input_shape)
    # vgg.save_weights('vgg16_weights.h5')
    # for layer in vgg.layers:
    #   layer.trainable= False
    # print(vgg.summary())
  #vgg.summary()
  # Encoder
  s1 = vgg.get_layer("block1_conv2").output   #256
  s2 = vgg.get_layer("block2_conv2").output   #128
  s3 = vgg.get_layer("block3_conv3").output   #64
  s4 = vgg.get_layer("block4_conv3").output   #32

  #Bridge
  b1 = vgg.get_layer("block5_conv3").output   #16

  #Decoder
  d1 = decoder_block(drop, b1, s4, 256)
  d2 = decoder_block(drop, d1, s3, 128)
  d3 = decoder_block(drop, d2, s2, 64)
  d4 = decoder_block(drop, d3, s1, 32)
  
  outputs = layers.Conv2D(1,(1,1), padding="same", activation="sigmoid")(d4)

  model = Model(inputs,outputs,name='Joan_1')
  return model