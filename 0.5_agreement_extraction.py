import numpy as np
import pandas as pd

def agreement(dataset_1, dataset_2):
  # Obtain the images common to both the datasets
  common = []
  images = []
  masks = []

  print("Identifying common images in both datasets...")
  for i in list(dataset_2['Img_name']):
      if i in list(dataset_1['Img_name']):
          common.append(i)
  print("Identification complete")
  print("Number of common images: ",len(common))


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
    image = ut.read_image(img_path_1,img_size)
    # images.append(img_path_1)

    msk_path_1 = '/content/masks/' + msk1 + '.png'
    mask1 = ut.read_image(msk_path_1,img_size,color_scale='gray')
    # masks.append(msk_path_1)

    msk_path_2 = '/content/masks/' + msk2 + '.png'
    mask2 = ut.read_image(msk_path_2,img_size,color_scale='gray')
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