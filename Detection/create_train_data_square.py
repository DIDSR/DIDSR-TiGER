import cv2
import numpy as np

def create_train_data_square(img,bboxes,scale, size):
    # this function creates a binary mask from the image and the TILs locations.
    
    # inputs: img: numpy araye of size (N,N,3)
    # bboxes : TILs locations, numpy array of size (M,2)
    # scale: a scalar value to scale the binary mask to different patch sizes. For our task scale is chosen to be 1.
    # size is the square mask size. Each TILs location centriod point will be enlarged into a square of size "size".
    
    #output: binary mask of the TILs locations. Coordinates of the TILs locations on the binary mask.
    image_size_scaled = int(img.shape[0]/scale)
    mask_bg_fg = np.zeros((image_size_scaled,image_size_scaled,1))
    mask_coords = np.zeros((image_size_scaled,image_size_scaled,2))
    bboxes = bboxes/scale
    N = 0
    size = int(size/scale)
    for i in range(bboxes.shape[0]):    
        N += 1
        cv2.rectangle(mask_bg_fg, (int(bboxes[i,0]-size/2), int(bboxes[i,1]-size/2),size,size), N, -1)
        I = np.argwhere(mask_bg_fg == N)
        for j in range(I.shape[0]):        
            mask_coords[I[j,0],I[j,1],0] = bboxes[i,1]-I[j,0]
            mask_coords[I[j,0],I[j,1],1] = bboxes[i,0]-I[j,1]
    mask_bg_fg[mask_bg_fg != 0]=1
    mask_bg_fg = np.reshape(mask_bg_fg, (mask_bg_fg.shape[0],mask_bg_fg.shape[1],1))
    return mask_bg_fg.astype('uint8'), mask_coords.astype('uint8')
