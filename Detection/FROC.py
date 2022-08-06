import numpy as np
from scipy.spatial.distance import cdist

def count_tp_fp_fn(pred,gt,prob_threshold,hit_distance):   
    if pred.shape[0]>0:
        pred = pred[np.argwhere(pred[:,2]>=prob_threshold)[:,0],:2]        
    if pred.shape[0]>0:    
        if gt.shape[0]>0:
            hit = cdist(pred[:,:2],gt)
            hit[hit<=hit_distance] = 1
            hit[hit!=1] = 0        
            sum_1 = np.sum(hit,0)
            sum_2 = np.sum(hit,1)
            fp = sum(sum_2 ==0)
            fn = sum(sum_1 ==0)
            tp = sum(sum_1 !=0)
        else:
            tp = 0
            fn = 0
            fp = pred.shape[0]
    else:
        tp = 0
        fp = 0
        fn = gt.shape[0]
    return tp, fp, fn

######## Calculating TP,FP,FN for each Patch, Add them all and then calculate Sen,FP-Per-Patch

def FROC(pred,gt,confidence_thresholds,hit_distance):
    sens = []
    fp_img = []
    for threshold in confidence_thresholds:
        tps = []
        fps = []
        fns = []
        for N in range(len(gt)):            
            tp, fp, fn = count_tp_fp_fn(pred[N],gt[N],threshold,hit_distance)
            tps.append(tp)
            fps.append(fp)
            fns.append(fn)
        tps = sum(tps)
        fp_per_img = np.mean(fps)
        fps = sum(fps)
        fns = sum(fns)
        if tps!=0:
            sens.append(tps/(tps+fns))
        else:
            sens.append(0)
        fp_img.append(fp_per_img)
    return sens, fp_img

######## Calculating Sens,FP_per_Image Patch Wise and then taking the average

# def FROC(pred,gt,confidence_thresholds,hit_distance):
#     sens = []
#     fp_img = []
#     for threshold in confidence_thresholds:        
#         sensitivity = []
#         fp_per_image = [] 
#         for N in range(len(gt)):            
#             tp, fp, fn = count_tp_fp_fn(pred[N],gt[N],threshold,hit_distance)
#             if tp!=0:            
#                 sensitivity.append(tp/(tp+fn))
#             if tp==0 and fp!=0 or fn!=0:
#                 sensitivity.append(0)                     
#             fp_per_image.append(fp)
#         sens.append(np.mean(sensitivity))
#         fp_img.append(np.mean(fp_per_image))
#     return sens, fp_img
