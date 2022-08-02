import os
import cv2
from wholeslidedata.image.wholeslideimage import WholeSlideImage
import numpy as np
from find_tcga_rotation import find_tcga_rotation
####################################################
def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def extract_tcga_data(data_path):
    dir_TIFF_images_WSIROIS = data_path+'/wsirois/wsi-level-annotations/images/'
    dir_PNG_masks = data_path+'/wsirois/roi-level-annotations/tissue-bcss/masks/'    
    dir_PNG_images = data_path+'/wsirois/roi-level-annotations/tissue-bcss/images/'
    dir_XML = data_path+'/wsirois/wsi-level-annotations/annotations-tissue-bcss-xmls/'    
   
    imgs_names = os.listdir(dir_TIFF_images_WSIROIS)
    imgs_names.sort()
    msks_names = os.listdir(dir_PNG_masks)
    msks_names.sort()
    imgs_names = [i for i in imgs_names if i.startswith('TCGA')]    
    png_names = os.listdir(dir_PNG_images)
    png_names.sort()
    png_names = png_names[:-1]    
    
    imgs_msks_names_masks_coorindates = []
    for i in range(len(imgs_names)):
        temp = imgs_names[i][:-4]
        msk_names = [j for j in msks_names if temp in j[:60]]
        msk_coordinates = [j[62:-5] for j in msk_names]
        msk_coordinates = [j.split(', ') for j in msk_coordinates]
        msk_coordinates = [[int(k) for k in j] for j in msk_coordinates]        
        imgs_msks_names_masks_coorindates.append([imgs_names[i], msk_names, msk_coordinates])  
    del temp, i, msk_names,msk_coordinates

    imgs = []
    msks = []
    for i in range(len(imgs_msks_names_masks_coorindates)):      
        tif_name = imgs_msks_names_masks_coorindates[i][0]
        tif_img = WholeSlideImage(dir_TIFF_images_WSIROIS+tif_name)
        print(tif_name)        
        msk_name = imgs_msks_names_masks_coorindates[i][1][0]
        x_min_bound = imgs_msks_names_masks_coorindates[i][2][0][0]
        y_min_bound = imgs_msks_names_masks_coorindates[i][2][0][1]
        x_max_bound = imgs_msks_names_masks_coorindates[i][2][0][2]
        y_max_bound = imgs_msks_names_masks_coorindates[i][2][0][3]
        img = tif_img.get_patch(x_min_bound, y_min_bound,x_max_bound-x_min_bound, y_max_bound-y_min_bound, tif_img.spacings[0], center = False)
        msk = cv2.imread(dir_PNG_masks+msk_name)[:,:,0]
        angle,x_min,x_max,y_min,y_max = find_tcga_rotation(dir_PNG_images, dir_XML, msk_name)
        if angle !=0:
            img_rotated = rotate_image(img, -angle) 
            img_rotated = img_rotated[y_min:y_max,x_min:x_max]
            msk_rotated = rotate_image(msk, -angle) 
            msk_rotated = msk_rotated[y_min:y_max,x_min:x_max]
        else:
            img_rotated = img
            msk_rotated = msk
        imgs.append(img_rotated)
        msks.append(msk_rotated) 
        
    return imgs, msks