from sklearn.model_selection import train_test_split
import tensorflow as tf


def preprocessing(img_path, mak_path):
    car_img = tf.io.read_file(img_path) 
    car_img = tf.image.decode_jpeg(car_img, channels=3)
    car_img = tf.image.central_crop(car_img, central_fraction=0.85)
    car_img = tf.image.resize(car_img, IMG_SIZE)
    car_img = tf.cast(car_img, tf.float32) / 255.0
    
    mask_img = tf.io.read_file(mak_path)
    mask_img = tf.image.decode_jpeg(mask_img, channels=3)
    mask_img = tf.image.central_crop(mask_img, central_fraction=0.85)
    mask_img = tf.image.resize(mask_img, IMG_SIZE)
    mask_img = mask_img[:,:,:1]    
    mask_img = tf.math.sign(mask_img)

    return car_img, mask_img

def create_dataset(df, train = False):
        ds = tf.data.Dataset.from_tensor_slices((df["image_path"].values, df["mask_path"].values))
        ds = ds.map(preprocessing, tf.data.AUTOTUNE)    
        return ds

def prepare_data(df, BATCH_SIZE, test1, test2):
    #preparing data 
    train = None
    valid = None
    test = None

    BUFFER_SIZE = 1000
    train_df, test_df = train_test_split(df, random_state=0, test_size=test1)
    valid_df, test_df = train_test_split(test_df, random_state=0, test_size=test2)

    train = create_dataset(train_df, train = True)
    valid = create_dataset(valid_df)
    test = create_dataset(test_df)

    train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    valid_dataset = valid.batch(BATCH_SIZE)
    test_dataset = test.batch(BATCH_SIZE)

    return train_dataset, valid_dataset, test_dataset, len(train_df)

