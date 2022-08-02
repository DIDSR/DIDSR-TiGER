from extract_names_TIFF_files import extract_names_TIFF_files
from extract_roi_polygon_bbox_XML_file import extract_roi_polygon_bbox_XML_file
from extract_patch_tils_from_tiff import extract_patch_tils_from_tiff
import numpy as np
import matplotlib.pyplot as plt
########################## load TCGA images and TILs ##########################
def extract_tcga_data(path):
    dir_TIFF_images_WSIROIS = path+'/wsirois/wsi-level-annotations/images/'
    dir_TIFF_masks_WSIROIS = path+'/wsirois/wsi-level-annotations/images/tissue-masks/'
    dir_XML_WSIROIS = path+'/wsirois/wsi-level-annotations/annotations-tissue-cells-xmls/'
    TIFF_images_names = extract_names_TIFF_files(dir_TIFF_images_WSIROIS)
    imgs = []
    points = []
    imgs_reserve = []
    points_reserve = []
    np.random.seed(50)
    for name in TIFF_images_names['TCGA']:
        print(name)
        roi_polygon_bbox = extract_roi_polygon_bbox_XML_file(dir_XML_WSIROIS, name)
        if len(roi_polygon_bbox)>1:
            L = int(np.round(len(roi_polygon_bbox)*10/100))
        else:
            L = 0
        print(L)
        idx = np.random.permutation(np.arange(0,len(roi_polygon_bbox)))
        idx_reserve = idx[:L]
        idx = idx[L:]      
        for i in idx:
            img, bboxes = extract_patch_tils_from_tiff(dir_TIFF_images_WSIROIS, name, roi_polygon_bbox[i], size_image = 500)
            # plt.cla()
            # plt.imshow(img)
            # if bboxes.shape[0]>0:
            #     plt.scatter(bboxes[:,0],bboxes[:,1], s = 10, c = 'r')
            # plt.draw()
            # plt.pause(0.001)
            imgs.append(img)
            points.append(bboxes)
        for i in idx_reserve:
            img, bboxes = extract_patch_tils_from_tiff(dir_TIFF_images_WSIROIS, name, roi_polygon_bbox[i], size_image = 500)
            # plt.cla()
            # plt.imshow(img)
            # if bboxes.shape[0]>0:
            #     plt.scatter(bboxes[:,0],bboxes[:,1], s = 10, c = 'r')
            # plt.draw()
            # plt.pause(0.001)
            imgs_reserve.append(img)
            points_reserve.append(bboxes)        
    del name, roi_polygon_bbox, i, img, bboxes, dir_TIFF_images_WSIROIS, dir_TIFF_masks_WSIROIS, dir_XML_WSIROIS, TIFF_images_names, L, idx, idx_reserve

    return imgs, points, imgs_reserve, points_reserve