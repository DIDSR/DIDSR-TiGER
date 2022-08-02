from wholeslidedata.annotation.wholeslideannotation import WholeSlideAnnotation
import numpy as np
import cv2
import math

def rotate_point(origin, point, angle):
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def find_tcga_rotation(dir_imgs, dir_XML, name):
    PNG_img = cv2.imread(dir_imgs+name)
    shapes = PNG_img.shape
    
    xml_info = WholeSlideAnnotation(dir_XML+name[:60]+'.xml')
    
    annotations = xml_info.sampling_annotations
    
    labels_values = xml_info.labels.values
    labels_names = xml_info.labels.names
    annotations = xml_info.sampling_annotations
    
    rois = []
    roi_label = labels_names.index('roi')
    roi_label = labels_values[roi_label]
    for i in annotations:        
        if i.label.value == roi_label:
            temp = i.wkt[10:-2].split(', ')        
            temp = [i.split(' ') for i in temp]
            temp = np.array([[int(float(j)) for j in i] for i in temp])
            rois.append(temp)
    
    roi = rois[0]
    shapes = PNG_img.shape
    move_x = (max(roi[:,0])+min(roi[:,0]))/2-shapes[1]/2
    move_y = (max(roi[:,1])+min(roi[:,1]))/2-shapes[0]/2
    roi[:,0] = roi[:,0]-move_x
    roi[:,1] = roi[:,1]-move_y

    for j in range(roi.shape[0]):
        if roi[j,0]==0 and roi[j,1]!=0:
            x1,y1 = roi[j,:]
        if roi[j,0]!=0 and roi[j,1]==0:
            x2,y2 = roi[j,:]    
    angle = np.degrees(np.arctan((y2-y1)/(x2-x1)))     
    angle_image = np.degrees(np.arctan((shapes[0]-0)/(shapes[1]-0)))  

    if abs(np.round(angle_image)) != abs(np.round(angle)):        
        if abs(angle)<45:
            angle = abs(angle)
        else:
            angle = -(90-abs(angle))  
        origin = [shapes[1]/2,shapes[0]/2]            
        rois_rotated = np.zeros((5,2))
        rois_rotated[0,:] = rotate_point(origin,roi[0,:],np.radians(angle))
        rois_rotated[1,:] = rotate_point(origin,roi[1,:],np.radians(angle))
        rois_rotated[2,:] = rotate_point(origin,roi[2,:],np.radians(angle))
        rois_rotated[3,:] = rotate_point(origin,roi[3,:],np.radians(angle))
        rois_rotated[4,:] = rois_rotated[0,:]
        
        x_min = int(min(rois_rotated[:,0]))
        x_max = int(max(rois_rotated[:,0]))
        y_min = int(min(rois_rotated[:,1]))
        y_max = int(max(rois_rotated[:,1]))  

        if x_min<0:
            x_min = 0
        if x_max>shapes[1]:
            x_max = shapes[1]
        if y_min<0:
            y_min = 0
        if y_max>shapes[0]:
            y_max = shapes[0]    

    else:
        angle = 0
        x_min = 0
        y_min = 0
        x_max = 0
        y_max = 0      
    return angle,x_min,x_max,y_min,y_max