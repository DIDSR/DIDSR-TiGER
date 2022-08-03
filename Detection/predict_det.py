import numpy as np
from scipy.spatial.distance import cdist
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def crop_img(img,size,step):
    cropped_img = [] 
    cropped_coords = []
    size_x = img.shape[1]
    size_y = img.shape[0]
    x = np.arange(0,size_x,step)
    y = np.arange(0,size_y,step)    
    for idx,val in enumerate(x):
        if val+size>=size_x:
            val = size_x-size
            x[idx] = val    
    for idx,val in enumerate(y):
        if val+size>=size_y:
            val = size_y-size
            y[idx] = val 
    x = np.unique(x)
    y = np.unique(y)
    if len(img.shape)>2:
        for i in x:
            for j in y:
                cropped_img.append(img[j:j+size,i:i+size,:])    
                cropped_coords.append([i,j])
    elif len(img.shape)==2:
        for i in x:
            for j in y:
                cropped_img.append(img[j:j+size,i:i+size])    
                cropped_coords.append([i,j])        
    return cropped_img, cropped_coords 

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

def pred_all_detc(cell_det_model, image, images, locations, patch_size):
    predicted = cell_det_model.predict(images, batch_size = 32, verbose = 1)  
    predicted = predicted[:,:,:,0]
    
    predicted_all_max = np.zeros((image.shape[0],image.shape[1]))    
    for i in range(len(locations)):        
        temp = np.zeros((patch_size,patch_size,2))
        temp[:,:,0] = predicted_all_max[locations[i][1]:locations[i][1]+patch_size,locations[i][0]:locations[i][0]+patch_size]
        temp[:,:,1] = predicted[i]
        # temp[:,:,1] = predicted_all[i]
        temp = np.max(temp, axis = 2)
        predicted_all_max[locations[i][1]:locations[i][1]+patch_size,locations[i][0]:locations[i][0]+patch_size] = temp       
    return predicted_all_max

def extract_predictions_from_ROI(cell_det_model, image, patch_size):   
    ###cell detection###
    crop_size = patch_size
    crop_stride = int(patch_size/2)
    images, locations =  crop_img(image,crop_size,crop_stride)
    images = np.array(images)    
    predicted_detections = pred_all_detc(cell_det_model, image, images, locations, patch_size)    
    distance_threshold = 8
    confidence_threshold = 0.1
    predicted_detections_points = extract_predictions(predicted_detections, confidence_threshold = confidence_threshold)
    predicted_detections_points = non_max_supression_distance(predicted_detections_points,distance_threshold = distance_threshold) 

    return predicted_detections_points

def predict_det(cell_det_model, image):
    image = (image/255.0).astype('float32')
    pred_cells = extract_predictions_from_ROI(cell_det_model, image, patch_size = 256)
    return pred_cells
