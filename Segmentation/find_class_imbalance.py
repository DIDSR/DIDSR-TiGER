import numpy as np

def find_class_imbalance(masks):    
    unique_vals = [np.unique(i, return_counts = True) for i in masks]
    
    classes = np.unique(np.concatenate([i[0] for i in unique_vals]))
    
    classes_counts = np.zeros((len(classes),1))
    for i in unique_vals:        
        for j in range(len(classes)):
            if j in i[0]:
                index_val = list(i[0]).index(j)
                classes_counts[j] += i[1][index_val]
                
    classes_counts = classes_counts/np.sum(classes_counts)
    return np.round(classes_counts,3)*100