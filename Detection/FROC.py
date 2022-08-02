import numpy as np
from scipy.spatial.distance import cdist

def count_tp_fp_fn(pred,gt,score,distance_threshold):   
    if pred.shape[0]>0:
        pred = pred[np.argwhere(pred[:,2]>=score)[:,0],:2]
        if gt.shape[0]>0 and pred.shape[0]>0:  
            hit = cdist(pred[:,:2],gt)
            hit[hit<=distance_threshold] = 1
            hit[hit!=1] = 0        
            sum_1 = np.sum(hit,0)
            sum_2 = np.sum(hit,1)
            false_positives = sum(sum_2 ==0)
            false_negatives = sum(sum_1 ==0)
            true_positives = sum(sum_1 !=0)
        else:
            true_positives = 0
            false_negatives = 0
            false_positives = pred.shape[0]
    else:
        true_positives = 0
        false_negatives = 0
        false_positives = pred.shape[0]
    return true_positives, false_positives, false_negatives

def FROC(pred,gt,confidence_thresholds,distance_threshold):
    sens = []
    fp_img = []
    for i in confidence_thresholds:        
        print(i)
        sensitivity = []
        fp_per_image = []        
        for N in range(len(gt)):            
            true_positives, false_positives, false_negatives = count_tp_fp_fn(pred[N],gt[N],i,distance_threshold)
            if true_positives!=0:            
                sensitivity.append(true_positives/(true_positives+false_negatives))
            fp_per_image.append(false_positives)
        sens.append(np.mean(sensitivity))
        fp_img.append(np.mean(fp_per_image))
    return sens, fp_img