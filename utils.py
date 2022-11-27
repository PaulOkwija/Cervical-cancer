import pandas as pd
import cv2
from PIL import Image, ImageDraw
import numpy as np

def read_image(img_path, img_size=[512,512], color_scale='rgb'):
    ''' Returns a numpy array of the image '''
    image = cv2.imread(img_path)
    image = Image.fromarray(image)
    image = image.resize((img_size[0], img_size[1]))
    image = np.array(image, dtype=np.uint8)

    if color_scale == 'gray':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.expand_dims(image, axis=2)
    elif color_scale == 'rgb':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pass    

    return image

def IOU(y_true,y_test):
  inter = np.logical_and(y_true,y_test)
  union = np.logical_or(y_true,y_test)
  iou_score = np.sum(inter)/np.sum(union)
  return iou_score

def Dice(y_true,y_test):
  inter = np.logical_and(y_true,y_test)
  union = np.logical_or(y_true,y_test)
  dice_score = (2* np.sum(inter))/(np.sum(union) + np.sum(inter))
  return dice_score

def prepare_dataset(csv_file):
  dataset = pd.read_csv(csv_file)
  dataset = dataset.drop_duplicates(subset=dataset.columns[0])
  dataset = dataset.reset_index(drop=True)
  print("Shape of dataset from {} file: ".format(csv_file), dataset.shape)
  return dataset


def extract_diff_common(dataset_1, dataset_2):
  col = dataset_1.columns[0]
  common = list(
                set(dataset_1['{}'.format(col)].values) & set(dataset_2['{}'.format(col)].values)
                )
  extra_1 = list(
                set(dataset_1['{}'.format(col)].values) - set(dataset_2['{}'.format(col)].values)
                )
  extra_2 = list(
              set(dataset_2['{}'.format(col)].values) - set(dataset_1['{}'.format(col)].values)
                )
  print("Common images: ", len(common), 
        "\nDifferent images(dataset_1 only): ", len(extra_1),
        "\nDifferent images(dataset_2 only): ", len(extra_2))
  return common, extra_1, extra_2  


def extract_high_agreement_images(dataset_1, dataset_2,common):
    images = []
    masks = []

# Obtain the common images, masks and their respective paths from both datasets
    print("Extracting images and masks")
    count = 1
    for item in common:
        ind1 = dataset_1[dataset_1['Img_name']==item].index[0]
        ind2 = dataset_2[dataset_2['Img_name']==item].index[0]
        img, msk1, msk_x = dataset_1.iloc[ind1]
        _img, msk2 = dataset_2.iloc[ind2]

        print("Extracting image_{} and respective masks...".format(count))
        img_path_1 = '/content/images/' + img
        image = read_image(img_path_1,img_size)
        # images.append(img_path_1)

        msk_path_1 = '/content/masks/' + msk1 + '.png'
        mask1 = read_image(msk_path_1,img_size,color_scale='gray')
        # masks.append(msk_path_1)

        msk_path_2 = '/content/masks/' + msk2 + '.png'
        mask2 = read_image(msk_path_2,img_size,color_scale='gray')
        # masks.append(msk_path_1)

        print("Computing level of agreement...:")
        IOU_agg = IOU(mask1, mask2)
        Dice_agg = Dice(mask1, mask2)
        avg = (IOU_agg + Dice_agg)/2
        print("IOU:",IOU_agg,"Dice:",Dice_agg,"Avg_aggrement:",avg)

        # Collect the images with a high level of aggrement between two reviewers
        if avg>=0.5:
            images.append(img_path_1)
            masks.append(msk_path_1)

        count = count + 1

    print("Number of images collected:", len(images))
    print("#########################################")

    print("Generating final dataframe...")    
  # Generating a dataframe with images whose agreement is greater than 0.5
    image_and_mask = {'image_path':images, 'mask_path':masks}
    df = pd.DataFrame(image_and_mask)

    return df