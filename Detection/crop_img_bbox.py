import numpy as np

def crop_img_bbox(img,bboxes,size,step):
    bboxes = np.array(bboxes)
    cropped_img = []    
    cropped_bboxes = []
    size_x = img.shape[1]
    size_y = img.shape[0]
    x = np.arange(0,size_x,step)
    y = np.arange(0,size_y,step)
    x = [i for i in x if i+size<=size_x]
    y = [i for i in y if i+size<=size_y]
    x.append(size_x-size)
    y.append(size_y-size)
    x = np.unique(x)
    y = np.unique(y)
    for i in x:
        for j in y:
            cropped_img.append(img[j:j+size,i:i+size,:])           
            if bboxes.shape[0]>0:
                I_x = np.logical_and(bboxes[:,0]>i, bboxes[:,0]<i+size)
                I_y = np.logical_and(bboxes[:,1]>j, bboxes[:,1]<j+size)
                I = np.logical_and(I_x,I_y)
                wanted_x = bboxes[I,0]-i
                wanted_y = bboxes[I,1]-j
                wanted = np.zeros((wanted_x.shape[0],2))
                wanted[:,0] = wanted_x
                wanted[:,1] = wanted_y
                cropped_bboxes.append(wanted)
            else:
                cropped_bboxes.append(np.array([]))
    return cropped_img,cropped_bboxes