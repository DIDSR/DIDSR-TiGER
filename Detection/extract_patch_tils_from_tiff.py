import numpy as np
from wholeslidedata.image.wholeslideimage import WholeSlideImage

def extract_patch_tils_from_tiff(dir_TIFF_images, name, roi_polygon_bbox, size_image = 128):
    roi = roi_polygon_bbox[0][0]
    bboxes = roi_polygon_bbox[2]
    bboxes = [i[0] for i in bboxes]      
  
    x_min_bound = min(roi[:,0])
    x_max_bound = max(roi[:,0])
    y_min_bound = min(roi[:,1])
    y_max_bound = max(roi[:,1])
    h = y_max_bound-y_min_bound
    w = x_max_bound-x_min_bound
    
    roi_center_x = (x_max_bound+x_min_bound)/2
    roi_center_y = (y_max_bound+y_min_bound)/2
    
    if w<size_image:
        # delta_w = size_image-w
        # rand_w = np.random.randint(0,delta_w)      
        # x_min_bound = x_min_bound-delta_w+rand_w
        # x_min_bound = np.round(x_min_bound)
        # x_max_bound = x_min_bound+size_image
        x_min_bound = roi_center_x-size_image/2
        x_max_bound = roi_center_x+size_image/2
    if h<size_image:
        # delta_h = size_image-h
        # rand_h = np.random.randint(0,delta_h)   
        # y_min_bound = y_min_bound-delta_h+rand_h       
        # y_min_bound = np.round(y_min_bound)
        # y_max_bound = y_min_bound+size_image
        y_min_bound = roi_center_y-size_image/2
        y_max_bound = roi_center_y+size_image/2
        
    tif_img = WholeSlideImage(dir_TIFF_images+name)
    img = tif_img.get_patch(x_min_bound, y_min_bound,x_max_bound-x_min_bound, y_max_bound-y_min_bound, tif_img.spacings[0], center = False)  
    
    if len(bboxes)>0:
        for i in bboxes:
            i[:,0] = i[:,0]-x_min_bound
            i[:,1] = i[:,1]-y_min_bound 
        bboxes = [[(min(i[:,0])+max(i[:,0]))/2,(min(i[:,1])+max(i[:,1]))/2] for i in bboxes]
        bboxes = np.array(bboxes)
    else:
        bboxes = np.array([])
    return img, bboxes