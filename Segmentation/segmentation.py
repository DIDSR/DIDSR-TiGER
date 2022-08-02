import numpy as np

#### Input the location of the TIGER data in the loca disk######
data_path = ''

##### Loading TCGA images and masks #######
from extract_tcga_data import extract_tcga_data
imgs_1, msks_1 = extract_tcga_data(data_path)

##### Loading TC images and masks #######
from extract_tc_data import extract_tc_data
imgs_2, msks_2 = extract_tc_data(data_path)

##### Loading JB images and masks #######
from extract_jb_data import extract_jb_data
imgs_3, msks_3 = extract_jb_data(data_path)

##### Relabing the mask values to 0,1,2 #######
from change_masks import change_masks
msks_1 = [change_masks(i) for i in msks_1]  
msks_2 = [change_masks(i) for i in msks_2]  
msks_3 = [change_masks(i) for i in msks_3] 

########## Combining all images and masks ########
########## Combining all images and masks ########
########## Combining all images and masks ########
images = []
for i in imgs_1:
    images.append(i)
for i in imgs_2:
    images.append(i)
for i in imgs_3:
    images.append(i) 
masks = []
for i in msks_1:
    masks.append(i)
for i in msks_2:
    masks.append(i)
for i in msks_3:
    masks.append(i)   
del imgs_1, imgs_2, imgs_3, msks_1, msks_2, msks_3, i

########## Extracting Patches from images and masks ########
########## Extracting Patches from images and masks ########
########## Extracting Patches from images and masks ########

# temporairly changing mask values of 0 to 3 for patch extraction
for i in range(len(masks)):
    masks[i][masks[i]==0] = 3

from crop_img_msk import crop_img_msk
patches = []
patches_masks = []
for i in range(len(images)):
    img_cropped,msk_cropped = crop_img_msk(images[i], masks[i], size = 256, stride = 256)
    for j in img_cropped:
        patches.append(j)
    for j in msk_cropped:
        patches_masks.append(j)  
del img_cropped,msk_cropped,i,j


from extract_patches_with_angle import extract_patches_with_angle

patches_rotated_one = []
patches_rotated_one_masks = []
for i in range(len(images)):
    img_rotated, msk_rotated = extract_patches_with_angle(images[i],masks[i], size = 256, stride = 256, angle = 45)
    print(i,len(img_rotated))
    for j in img_rotated:
        patches_rotated_one.append(j)
    for j in msk_rotated:
        patches_rotated_one_masks.append(j)  
    del img_rotated, msk_rotated
del i, j

patches_rotated_two = []
patches_rotated_two_masks = []
for i in range(len(images)):
    img_rotated, msk_rotated = extract_patches_with_angle(images[i],masks[i], size = 256, stride = 256, angle = -45)
    print(i,len(img_rotated))
    for j in img_rotated:
        patches_rotated_two.append(j)
    for j in msk_rotated:
        patches_rotated_two_masks.append(j)  
    del img_rotated, msk_rotated, j
del i

#combining all patches
for i in patches_rotated_one:
    patches.append(i)
for i in patches_rotated_one_masks:
    patches_masks.append(i)  
del i, patches_rotated_one, patches_rotated_one_masks

for i in patches_rotated_two:
    patches.append(i)
for i in patches_rotated_two_masks:
    patches_masks.append(i)  
del i, patches_rotated_two, patches_rotated_two_masks

#changing the mask values from 3 back to 0
for i in range(len(patches_masks)):
    patches_masks[i][patches_masks[i]==3] = 0
del i
    
################## Shuffling Patches #######################
################## Shuffling Patches #######################
################## Shuffling Patches #######################
idx = np.random.RandomState(seed=42).permutation(np.arange(0,len(patches)))
patches = [patches[i] for i in idx]
patches_masks = [patches_masks[i] for i in idx]  
del idx

################## Augmenting Patches #######################  
################## Augmenting Patches #######################  
################## Augmenting Patches #######################  
from aug_imgs_msks_d4 import aug_imgs_msks_d4
patches, patches_masks = aug_imgs_msks_d4(patches,patches_masks)


################# Finding Class Imbalance ###################
################# Finding Class Imbalance ###################
################# Finding Class Imbalance ###################
from find_class_imbalance import find_class_imbalance
find_class_imbalance(patches_masks) 


########################## Training Seg Model ##########################
########################## Training Seg Model ##########################
########################## Training Seg Model ##########################
from tensorflow.keras.utils import to_categorical
def image_mask_generator(imgs, msks, batch_size):        
    L = len(imgs)
    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < L:
            limit = min(batch_end, L)              
            temp_imgs = []
            temp_msks = []            
            for i in range(batch_start,limit):            
                temp_imgs.append(imgs[i])
                temp_msks.append(msks[i])
            temp_imgs = [preprocess_input(i) for i in temp_imgs]                
            temp_msks = [to_categorical(i, num_classes = 3) for i in temp_msks]
            temp_imgs = np.array(temp_imgs)
            temp_msks = np.array(temp_msks)  
            yield (temp_imgs,temp_msks) #a tuple with two numpy arrays with batch_size samples
            batch_start += batch_size  
            batch_end += batch_size

import os
import random
import tensorflow
import segmentation_models as sm
from segmentation_model import model
model, preprocess_input = model(dropout_value=0.4)

class_weights = np.array([1.6, 1, 0.76])
dice_loss = sm.losses.DiceLoss(class_weights = class_weights)
focal_loss = sm.losses.CategoricalFocalLoss()    
total_loss = dice_loss + (1 * focal_loss)
metrics = sm.metrics.IOUScore()   
model.compile(tensorflow.keras.optimizers.Adam(learning_rate=0.0001), total_loss, metrics=metrics) 

epochs = 30
batch_size = 32
os.environ['PYTHONHASHSEED'] = str(12321)
random.seed(12321)
np.random.seed(12321)
tensorflow.random.set_seed(12321)
filepath = "inceptionv3_256_both_angle_dropout_point4_all-{epoch:02d}.hdf5"
my_callbacks = [    
    tensorflow.keras.callbacks.ModelCheckpoint(filepath, verbose=0, save_best_only=False, save_weights_only = True)
    ]    

model.fit(image_mask_generator(patches, patches_masks, batch_size),
          epochs = epochs,
          steps_per_epoch = len(patches)//batch_size,
          verbose = 1,          
          callbacks = my_callbacks)

