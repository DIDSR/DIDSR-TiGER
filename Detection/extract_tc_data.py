from extract_names_TIFF_files import extract_names_TIFF_files
from extract_roi_polygon_bbox_XML_file import extract_roi_polygon_bbox_XML_file
from extract_patch_tils_from_tiff import extract_patch_tils_from_tiff
import numpy as np
import matplotlib.pyplot as plt
####################################################
def extract_tc_data(path):
    dir_TIFF_images_WSIROIS = path+'/wsirois/wsi-level-annotations/images/'
    dir_TIFF_masks_WSIROIS = path+'/wsirois/wsi-level-annotations/images/tissue-masks/'
    dir_XML_WSIROIS = path+'/wsirois/wsi-level-annotations/annotations-tissue-cells-xmls/'
    TIFF_images_names = extract_names_TIFF_files(dir_TIFF_images_WSIROIS)
    imgs = []
    points = []
    np.random.seed(50)
    for name in TIFF_images_names['TC']:
        print(name)
        roi_polygon_bbox = extract_roi_polygon_bbox_XML_file(dir_XML_WSIROIS, name)         
        for i in roi_polygon_bbox:
            img, bboxes = extract_patch_tils_from_tiff(dir_TIFF_images_WSIROIS, name, i, size_image = 1)
            # plt.cla()
            # plt.imshow(img)
            # if bboxes.shape[0]>0:
            #     plt.scatter(bboxes[:,0],bboxes[:,1], s = 10, c = 'r')
            # plt.draw()
            # plt.pause(0.001)
            imgs.append(img)
            points.append(bboxes)          
    del name, roi_polygon_bbox, i, img, bboxes, dir_TIFF_images_WSIROIS, dir_TIFF_masks_WSIROIS, dir_XML_WSIROIS, TIFF_images_names    
    np.random.seed(50)
    idx = np.random.permutation(np.arange(0,len(imgs)))
    idx_reserve = idx[:5]
    idx = idx[5:]
    
    imgs_reserve = [imgs[i] for i in idx_reserve]
    points_reserve = [points[i] for i in idx_reserve]

    imgs = [imgs[i] for i in idx]
    points = [points[i] for i in idx]
    return imgs, points, imgs_reserve, points_reserve