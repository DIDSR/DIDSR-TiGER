import numpy as np
from scipy.spatial.distance import cdist

def count_tp_fp_fn(pred,gt,prob_threshold,hit_distance): 
    # This function counts the number of true-positives, false negatives and false positives in a detection taks. 
    
    # inputs: A numpy array of shape [N,3] for pred and gt. N is the number of detections. The first column is the x-position of the detections.
    # the second column is the y-position of the detections. the third column is the probability associated with detections.
    # The hit-distance is the maximum distance of a detection from the ground-truth to be counted as a true-positive, otherwsie, it will be counted as a false positive.
    # The prob_threshld is the threshold value in which detections with probabilities smaller than the threshold value will be discarded.
    
    #output: number of ture-positives, false positives, and false negatives.
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
    # this function calculates the average sensitivity and average number of fp_per_image.
    
    # inputs: pred and gt are lists containing detections from the ground truth and the predictions from the model. for example:
    # pred = [pred_1, pred_2, pred_3], gt = [gt_1, gt_2, gt_3]
    # pred_1 = np.array((N,3))  pred_2 = np.array((M,3))    pred_3 = np.array((P,3))
    # gt_1 = np.array((N,3))    gt_2 = np.array((M,3))      gt_3 = np.array((P,3))    
    # confidence_thresholds is the list of the probability thresholds in calculating the FROC curve. for example:
    # confidence_thresholds = np.linspace(0,1,40)    
    # hit-distance: is the maximum distance from the ground truth by which a detection will be counted as a true positive.
    
    #outputs: returns a list for sensitivies and fps_per_image for each of the threshold values.
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
