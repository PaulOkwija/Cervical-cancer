from sklearn.model_selection import train_test_split
import tensorflow as tf


def preprocessing(img_path, mak_path):
    IMG_SIZE = [512,512]
    car_img = tf.io.read_file(img_path) 
    car_img = tf.image.decode_jpeg(car_img, channels=3)
    # car_img = tf.image.central_crop(car_img, central_fraction=0.85)
    car_img = tf.image.resize(car_img, IMG_SIZE)
    car_img = tf.cast(car_img, tf.float32) / 255.0
    
    mask_img = tf.io.read_file(mak_path)
    mask_img = tf.image.decode_jpeg(mask_img, channels=3)
    # mask_img = tf.image.central_crop(mask_img, central_fraction=0.85)
    mask_img = tf.image.resize(mask_img, IMG_SIZE)
    mask_img = mask_img[:,:,:1]    
    mask_img = tf.math.sign(mask_img)

    return car_img, mask_img

def flip(image,mask):
    image = tf.image.random_flip_left_right(image)
    mask = tf.image.random_flip_left_right(mask)
    return image, mask

def rotate(image,mask):
    image = tf.image.rot90(image)
    mask = tf.image.rot90(mask)
    return image, mask

def create_dataset(df, train = False, aug = False):
    ds = tf.data.Dataset.from_tensor_slices((df["image_path"].values, df["mask_path"].values))
    ds = ds.map(preprocessing, tf.data.AUTOTUNE)
    if aug:
        ds2 = ds.map(flip)
        # ds3 = ds.map(crop)
        ds4 = ds.map(rotate)
        ds = ds.concatenate(ds2)
        # ds = ds.concatenate(ds3)
        ds = ds.concatenate(ds4)
    return ds
