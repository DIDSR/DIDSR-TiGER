import os
from wholeslidedata.annotation.wholeslideannotation import WholeSlideAnnotation
import numpy as np

def extract_roi_polygon_bbox_XML_file(dir_XML, name):
    xml_files = os.listdir(dir_XML)   
    xml_files.sort()
    if name[:-4]+'.xml' in xml_files:
        xml_info = WholeSlideAnnotation(dir_XML+name[:-4]+'.xml')
        
        labels_values = xml_info.labels.values
        labels_names = xml_info.labels.names
        roi_label = labels_names.index('roi')
        roi_label = labels_values[roi_label]
        if 'lymphocytes and plasma cells' in labels_names:
            point_lable = labels_names.index('lymphocytes and plasma cells')
            point_lable = labels_values[point_lable]
        else:
            point_lable = 10000
        annotations = xml_info.sampling_annotations
        
        polygons = []
        points = []
        rois = []
        for i in annotations:        
            if i.label.value == roi_label:
                rois.append([i,i.label.name])
            elif i.label.value == point_lable:
                points.append([i,i.label.name])
            else:
                polygons.append([i,i.label.name])         
        
        rois_polygons_bboxes = []
        for i in rois:
            temp = i[0].wkt[10:-2].split(', ')        
            temp = [i.split(' ') for i in temp]
            temp = np.array([[int(float(j)) for j in i] for i in temp])
            rois_polygons_bboxes.append([[temp, i[1]],[],[]])
            
        for i in range(len(rois)):    
            for j in range(len(polygons)):        
                # if rois[i][0].intersects(polygons[j][0]):
                temp = polygons[j][0].wkt[10:-2].split(', ')        
                temp = [i.split(' ') for i in temp]
                temp = np.array([[int(float(j)) for j in i] for i in temp])
                rois_polygons_bboxes[i][1].append([temp, polygons[j][1]])
            for j in range(len(points)):
                if rois[i][0].intersects(points[j][0]):
                    temp = points[j][0].wkt[10:-2].split(', ')        
                    temp = [i.split(' ') for i in temp]
                    temp = np.array([[int(float(j)) for j in i] for i in temp])
                    rois_polygons_bboxes[i][2].append([temp, points[j][1]])            
    else:
        rois_polygons_bboxes = []
    return rois_polygons_bboxes