import matplotlib.pyplot as plt
import numpy as np
########################## load TCGA, TC, JB images and TILs ##########################
path = '/home/arian/Desktop/ResearchComputer/TIGER Data'

from extract_tcga_data import extract_tcga_data
tcga_imgs, tcga_points, tcga_imgs_reserve, tcga_points_reserve = extract_tcga_data(path)
print('Number of TCGA Train Images = '+str(len(tcga_imgs)))
print('Number of TCGA Train Points = '+str(sum([i.shape[0] for i in tcga_points])))
print('Number of TCGA Test Images = '+str(len(tcga_imgs_reserve)))
print('Number of TCGA Test Points = '+str(sum([i.shape[0] for i in tcga_points_reserve])))

from extract_tc_data import extract_tc_data
tc_imgs, tc_points, tc_imgs_reserve, tc_points_reserve = extract_tc_data(path)
print('Number of TC Train Images = '+str(len(tc_imgs)))
print('Number of TC Train Points = '+str(sum([i.shape[0] for i in tc_points])))
print('Number of TC Test Images = '+str(len(tc_imgs_reserve)))
print('Number of TC Test Points = '+str(sum([i.shape[0] for i in tc_points_reserve])))

from extract_jb_data import extract_jb_data
jb_imgs, jb_points, jb_imgs_reserve, jb_points_reserve = extract_jb_data(path)
print('Number of JB Train Images = '+str(len(jb_imgs)))
print('Number of JB Train Points = '+str(sum([i.shape[0] for i in jb_points])))
print('Number of JB Test Images = '+str(len(jb_imgs_reserve)))
print('Number of JB Test Points = '+str(sum([i.shape[0] for i in jb_points_reserve])))

########################## Extracting Patches ##########################
# values for pach size of 256: [90,122,154]
patch_size = 256
patches_tcga = []
patches_tcga_points = []
for i in range(len(tcga_imgs)):
    img = tcga_imgs[i]
    tils = tcga_points[i]
    x = [90,122,154]
    y = [90,122,154]
    for i in x:
        for j in y:
            patches_tcga.append(img[j:j+patch_size,i:i+patch_size,:])           
            if tils.shape[0]>0:
                I_x = np.logical_and(tils[:,0]>i, tils[:,0]<i+patch_size)
                I_y = np.logical_and(tils[:,1]>j, tils[:,1]<j+patch_size)
                I = np.logical_and(I_x,I_y)
                wanted_x = tils[I,0]-i
                wanted_y = tils[I,1]-j
                wanted = np.zeros((wanted_x.shape[0],2))
                wanted[:,0] = wanted_x
                wanted[:,1] = wanted_y
                patches_tcga_points.append(wanted)
            else:
                patches_tcga_points.append(np.array([]))
del i, img, tils, x, y, j, I_x, I_y, I, wanted_x, wanted_y, wanted, patch_size
del tcga_imgs, tcga_points

from crop_img_bbox import crop_img_bbox
patches_tc = []
patches_tc_points = []
for i in range(len(tc_imgs)):
    img_cropped,bbox_cropped = crop_img_bbox(tc_imgs[i], tc_points[i], size = 256, step = 128)
    for i in img_cropped:
        patches_tc.append(i)
    for i in bbox_cropped:
        patches_tc_points.append(i)  
del img_cropped,bbox_cropped,i
del tc_imgs, tc_points

patches_jb = []
patches_jb_points = []
for i in range(len(jb_imgs)):
    img_cropped,bbox_cropped = crop_img_bbox(jb_imgs[i], jb_points[i], size = 256, step = 128)
    for i in img_cropped:
        patches_jb.append(i)
    for i in bbox_cropped:
        patches_jb_points.append(i)  
del img_cropped,bbox_cropped,i
del jb_imgs, jb_points

all_patches = []
for i in patches_tcga:
    all_patches.append(i)
for i in patches_tc:
    all_patches.append(i)   
for i in patches_jb:
    all_patches.append(i)     
del patches_tcga,patches_tc,patches_jb
    
all_points = []
for i in patches_tcga_points:
    all_points.append(i)
for i in patches_tc_points:
    all_points.append(i)   
for i in patches_jb_points:
    all_points.append(i)    
del i
del patches_tcga_points, patches_tc_points, patches_jb_points
print('Number of All Patches (After Crop) = '+str(len(all_patches)))
print('Number of All Patches Points (After Crop) = '+str(sum([i.shape[0] for i in all_points])))
####### Augmenting Patches #######
from aug_imgs_points import aug_imgs_points
aug_patches, aug_points = aug_imgs_points(all_patches, all_points, patch_size = 256)
print('Number of All Patches (After Aug) = '+str(len(aug_patches)))
print('Number of All Patches Points (After Aug) = '+str(sum([i.shape[0] for i in aug_points])))
del all_patches, all_points
####### Shuffle Patches #######
np.random.seed(50)
idx = np.random.permutation(np.arange(0,len(aug_patches)))
aug_patches = [aug_patches[i] for i in idx]
aug_points = [aug_points[i] for i in idx]
del idx
####### Split data for Train #######
train_imgs = aug_patches
train_points = aug_points
del aug_patches, aug_points
########################## Extracting Patches for Test Data ##########################
ROIs_reserve_imgs = []
for i in tc_imgs_reserve:
    ROIs_reserve_imgs.append(i)    
for i in jb_imgs_reserve:
    ROIs_reserve_imgs.append(i)
ROIs_reserve_imgs = [(i/255.0).astype('float32') for i in ROIs_reserve_imgs]

ROIs_reserve_points =[]
for i in tc_points_reserve:
    ROIs_reserve_points.append(i)    
for i in jb_points_reserve:
    ROIs_reserve_points.append(i)
del i
    
patch_size = 256
patches_tcga = []
patches_tcga_points = []
for i in range(len(tcga_imgs_reserve)):
    img = tcga_imgs_reserve[i]
    tils = tcga_points_reserve[i]
    x = [90,122,154]
    y = [90,122,154]
    for i in x:
        for j in y:
            patches_tcga.append(img[j:j+patch_size,i:i+patch_size,:])           
            if tils.shape[0]>0:
                I_x = np.logical_and(tils[:,0]>i, tils[:,0]<i+patch_size)
                I_y = np.logical_and(tils[:,1]>j, tils[:,1]<j+patch_size)
                I = np.logical_and(I_x,I_y)
                wanted_x = tils[I,0]-i
                wanted_y = tils[I,1]-j
                wanted = np.zeros((wanted_x.shape[0],2))
                wanted[:,0] = wanted_x
                wanted[:,1] = wanted_y
                patches_tcga_points.append(wanted)
            else:
                patches_tcga_points.append(np.array([]))
del i, img, tils, x, y, j, I_x, I_y, I, wanted_x, wanted_y, wanted, patch_size
del tcga_imgs_reserve, tcga_points_reserve

patches_tc = []
patches_tc_points = []
for i in range(len(tc_imgs_reserve)):
    img_cropped,bbox_cropped = crop_img_bbox(tc_imgs_reserve[i], tc_points_reserve[i], size = 256, step = 128)
    for i in img_cropped:
        patches_tc.append(i)
    for i in bbox_cropped:
        patches_tc_points.append(i)  
del img_cropped,bbox_cropped,i
del tc_imgs_reserve, tc_points_reserve

patches_jb = []
patches_jb_points = []
for i in range(len(jb_imgs_reserve)):
    img_cropped,bbox_cropped = crop_img_bbox(jb_imgs_reserve[i], jb_points_reserve[i], size = 256, step = 128)
    for i in img_cropped:
        patches_jb.append(i)
    for i in bbox_cropped:
        patches_jb_points.append(i)  
del img_cropped,bbox_cropped,i
del jb_imgs_reserve, jb_points_reserve

all_patches = []
for i in patches_tcga:
    all_patches.append(i)
for i in patches_tc:
    all_patches.append(i)   
for i in patches_jb:
    all_patches.append(i)     
del patches_tcga,patches_tc,patches_jb
    
all_points = []
for i in patches_tcga_points:
    all_points.append(i)
for i in patches_tc_points:
    all_points.append(i)   
for i in patches_jb_points:
    all_points.append(i)    
del i
del patches_tcga_points, patches_tc_points, patches_jb_points
print('Number of All Patches (After Crop) = '+str(len(all_patches)))
print('Number of All Patches Points (After Crop) = '+str(sum([i.shape[0] for i in all_points])))

from aug_imgs_points_randomly import aug_imgs_points_randomly
aug_patches, aug_points = aug_imgs_points_randomly(all_patches, all_points, patch_size = 256)
print('Number of All Patches (After Aug) = '+str(len(aug_patches)))
print('Number of All Patches Points (After Aug) = '+str(sum([i.shape[0] for i in aug_points])))
del all_patches, all_points

np.random.seed(50)
idx = np.random.permutation(np.arange(0,len(aug_patches)))
aug_patches = [aug_patches[i] for i in idx]
aug_points = [aug_points[i] for i in idx]
del idx
test_imgs = aug_patches
test_points = aug_points
del aug_patches, aug_points

test_imgs = [(i/255.0).astype('float32') for i in test_imgs]
test_imgs = np.array(test_imgs)

from create_train_data_square import create_train_data_square
square_size = 12
test_msks = [create_train_data_square(test_imgs[i],test_points[i], scale = 1, size = square_size) for i in range(len(test_points))]    
test_msks = [i[0] for i in test_msks]  
test_msks = [i.astype('float32') for i in test_msks]  
test_msks = np.array(test_msks)
########################## Training Det Model ##########################
batch_size = 32
epochs = 1

def imageLoader(imgs, points, batch_size):        
    L = len(imgs)
    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < L:
            limit = min(batch_end, L)              
            train_imgs = []
            train_bboxes = []            
            for i in range(batch_start,limit):            
                train_imgs.append(imgs[i])
                train_bboxes.append(points[i]) 
            train_imgs = [(i/255.0).astype('float32') for i in train_imgs]
            train_msks = [create_train_data_square(train_imgs[i],train_bboxes[i],scale = 1, size = square_size) for i in range(len(train_bboxes))]   
            train_msks = [i[0] for i in train_msks] 
            train_msks = [i.astype('float32') for i in train_msks] 
            train_imgs = np.array(train_imgs)
            train_msks = np.array(train_msks)            
            yield (train_imgs,train_msks) #a tuple with two numpy arrays with batch_size samples 
            batch_start += batch_size   
            batch_end += batch_size

import segmentation_models as sm
sm.set_framework('tf.keras')
from segmentation_models import Unet
from segmentation_models.losses import BinaryCELoss
model = Unet(backbone_name = 'inceptionv3', input_shape= (256,256,3), encoder_weights='imagenet')
model.compile('Adam', loss=BinaryCELoss(), metrics='accuracy')

for epoch in range(epochs):     
    idx = np.random.permutation(np.arange(0,len(train_imgs)))
    train_imgs = [train_imgs[i] for i in idx]
    train_points = [train_points[i] for i in idx]
    model.fit(imageLoader(train_imgs, train_points, batch_size),                            
              steps_per_epoch = len(train_imgs)//batch_size,                            
              epochs = 1, 
              verbose = 1)
    model.evaluate(test_imgs, test_msks, batch_size = batch_size, verbose = 1)    
del idx

model.save('./inceptionv3_square_12_epochs_1.h5')
########################## Test Model on Test Images ##########################  
from FROC import FROC

model.load_weights('./inceptionv3_square_12_epochs_1.h5')

predicted_masks = model.predict(test_imgs, batch_size = 32, verbose = 1)
predicted_masks = predicted_masks[:,:,:,0]

from scipy.spatial.distance import cdist
def extract_predictions(probabilities, confidence_threshold):    
    indices = np.meshgrid(np.arange(0,probabilities.shape[1]),np.arange(0,probabilities.shape[0]))
    indices_x = indices[0]
    indices_y = indices[1]
    indices_x = indices_x.reshape((probabilities.shape[0]*probabilities.shape[1],1))
    indices_y = indices_y.reshape((probabilities.shape[0]*probabilities.shape[1],1))    
    probabilities = probabilities.reshape((probabilities.shape[0]*probabilities.shape[1],1))    
    boxes_pred = np.concatenate((indices_x,indices_y,probabilities),axis = 1)
    boxes_pred = boxes_pred[np.argsort(boxes_pred[:, 2])[::-1]]
    boxes_pred = boxes_pred[boxes_pred[:,2]>=confidence_threshold,:]   
    return boxes_pred
def non_max_supression_distance(points, distance_threshold):
    log_val = np.ones(points.shape[0])
    wanted = []
    for i in range(points.shape[0]):
        if log_val[i]:
            hit = cdist(np.expand_dims(points[i,:2],0),points[:,:2])
            hit = np.argwhere(hit<=distance_threshold)
            log_val[hit] = 0
            wanted.append(points[i,:])
    wanted = np.array(wanted)  
    return wanted

distance_threshold = 10
confidence_threshold = 0.1
predicted_detections = []
for i in range(len(predicted_masks)):
    print(i)
    temp = extract_predictions(predicted_masks[i], confidence_threshold = confidence_threshold)
    predicted_detections.append(non_max_supression_distance(temp,distance_threshold = distance_threshold))
del temp

confidence_thresholds = np.linspace(0,1,40)
sensitivity, fp_per_image = FROC(predicted_detections,test_points,confidence_thresholds,distance_threshold = 8)
plt.scatter(fp_per_image,sensitivity, s = 20, c = 'b')
plt.xlabel('Average FP per patch')
plt.ylabel('Sensitivity')
           
N = 7
plt.subplot(141)
plt.imshow(test_imgs[N])
if test_points[N].shape[0]>0:
    plt.scatter(test_points[N][:,0], test_points[N][:,1], edgecolors = 'b', s = 100)
plt.subplot(142)
plt.imshow(test_imgs[N])
if predicted_detections[N].shape[0]>0: 
    plt.scatter(predicted_detections[N][:,0], predicted_detections[N][:,1], edgecolors = 'r', s = 50, facecolors = 'none')
plt.subplot(143)
plt.imshow(test_msks[N])
plt.subplot(144)
plt.imshow(predicted_masks[N])

########################## Test Model on ROI ##########################  
from predict_det import predict_det
from FROC import FROC

model.load_weights('./inceptionv3_square_12_epochs_1.h5')

ROIs_predictions = []
for i in ROIs_reserve_imgs:
    ROIs_predictions.append(predict_det(model, i))
del i

confidence_thresholds = np.linspace(0,1,40)
sensitivity, fp_per_image = FROC(ROIs_predictions,ROIs_reserve_points,confidence_thresholds,distance_threshold = 8)
plt.scatter(fp_per_image,sensitivity, s = 20, c = 'b')

N = 0
plt.imshow(ROIs_reserve_imgs[N])
if ROIs_reserve_points[N].shape[0]>0:
    plt.scatter(ROIs_reserve_points[N][:,0], ROIs_reserve_points[N][:,1], facecolor = 'none', edgecolors = 'b', s = 100, linewidths=2)
if ROIs_predictions[N].shape[0]>0:
    plt.scatter(ROIs_predictions[N][:,0], ROIs_predictions[N][:,1], facecolor = 'none', edgecolors = 'r', s = 50, linewidths=2)