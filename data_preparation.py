from sklearn.model_selection import train_test_split
import tensorflow as tf


def preprocessing(img_path, mak_path):
    IMG_SIZE = [512,512]
    cerv_img = tf.io.read_file(img_path) 
    cerv_img = tf.image.decode_jpeg(cerv_img, channels=3)
    # cerv_img = tf.image.central_crop(cerv_img, central_fraction=0.85)
    cerv_img = tf.image.resize(cerv_img, IMG_SIZE)
    cerv_img = tf.cast(cerv_img, tf.float32) / 255.0
    
    mask_img = tf.io.read_file(mak_path)
    mask_img = tf.image.decode_jpeg(mask_img, channels=3)
    # mask_img = tf.image.central_crop(mask_img, central_fraction=0.85)
    mask_img = tf.image.resize(mask_img, IMG_SIZE)
    mask_img = mask_img[:,:,:1]    
    mask_img = tf.math.sign(mask_img)

    return cerv_img, mask_img

def flipLR(image,mask):
    image = tf.image.flip_left_right(image)
    mask = tf.image.flip_left_right(mask)
    return image, mask

def rotate(image,mask):
    image = tf.image.rot90(image)
    mask = tf.image.rot90(mask)
    return image, mask

def flipUD(image,mask):
    image = tf.image.flip_up_down(image)
    mask = tf.image.flip_up_down(mask)
    return image, mask


def create_dataset(df, train = False, aug = False, batch=2):
    BATCH_SIZE = batch
    ds = tf.data.Dataset.from_tensor_slices((df["image_path"].values, df["mask_path"].values))
    ds = ds.map(preprocessing, tf.data.AUTOTUNE)
    if aug:
        ds2 = ds.map(flipLR)
        ds3 = ds.map(flipUD)
        ds4 = ds.map(rotate)
        ds = ds.concatenate(ds2)
        ds = ds.concatenate(ds3)
        ds = ds.concatenate(ds4)
        print("Augmented train size:",len(ds))

    BUFFER_SIZE = 1000

    if train:
        ds = ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    else:
        ds = ds.batch(BATCH_SIZE)

    return ds

def prepare_data(df, BATCH_SIZE, test1):
    #preparing data 
    train_df, valid_df = train_test_split(df, random_state=0, test_size=test1)
    # valid_df, test_df = train_test_split(test_df, random_state=0, test_size=test2)

    print("Dataset_split:",
            "\nTrain: ", len(train_df), 
            "\nValidation: ", len(valid_df))

    train_dataset = create_dataset(train_df, train = True, aug=True, batch = BATCH_SIZE)
    valid_dataset = create_dataset(valid_df, batch = BATCH_SIZE)
    # test = create_dataset(test_df, batch = BATCH_SIZE)

 
            # "\nTest: ", len(test_df))

    return train_dataset, valid_dataset, len(train_df)