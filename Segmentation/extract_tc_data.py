import os
import cv2
from wholeslidedata.image.wholeslideimage import WholeSlideImage
import numpy as np
####################################################
def extract_tc_data(data_path):
    dir_TIFF_images_WSIROIS = data_path+'/wsirois/wsi-level-annotations/images/'
    dir_PNG_masks = data_path+'/wsirois/roi-level-annotations/tissue-cells/masks/'
    imgs_names = os.listdir(dir_TIFF_images_WSIROIS)
    imgs_names.sort()
    msks_names = os.listdir(dir_PNG_masks)
    msks_names.sort()
    imgs_names = [i for i in imgs_names if i.startswith('TC_S01')]
    msks_names = [i for i in msks_names if i.startswith('TC_S01')]
    
    imgs_msks_names_masks_coorindates = []
    for i in range(len(imgs_names)):
        temp = imgs_names[i][:-4]
        msk_names = [j for j in msks_names if temp in j[:25]]
        msk_coordinates = [j[27:-5] for j in msk_names]
        msk_coordinates = [j.split(', ') for j in msk_coordinates]
        msk_coordinates = [[int(k) for k in j] for j in msk_coordinates]        
        imgs_msks_names_masks_coorindates.append([imgs_names[i], msk_names, msk_coordinates])  

    imgs = []
    msks = []
    for i in range(len(imgs_msks_names_masks_coorindates)):        
        tif_name = imgs_msks_names_masks_coorindates[i][0]
        tif_img = WholeSlideImage(dir_TIFF_images_WSIROIS+tif_name)
        print(tif_name)
        for j in range(len(imgs_msks_names_masks_coorindates[i][1])):
            msk_name = imgs_msks_names_masks_coorindates[i][1][j]
            x_min_bound = imgs_msks_names_masks_coorindates[i][2][j][0]
            y_min_bound = imgs_msks_names_masks_coorindates[i][2][j][1]
            x_max_bound = imgs_msks_names_masks_coorindates[i][2][j][2]
            y_max_bound = imgs_msks_names_masks_coorindates[i][2][j][3]
            imgs.append(tif_img.get_patch(x_min_bound, y_min_bound,x_max_bound-x_min_bound, y_max_bound-y_min_bound, tif_img.spacings[0], center = False))
            msks.append(cv2.imread(dir_PNG_masks+msk_name)[:,:,0])
    return imgs, msks