import numpy as np

def aug_imgs_points(patches, patches_points, patch_size = 128):
    # this function augments patches and detections by keeping the original patches, flipping patches left-right, flipping patches up-down, and transposing patches.
    
    # inputs: patches = a list of patches images: [patch_1, patch_2, patch_3, ....]
    # patches_points = a list of TILs for patches: [points_1, points_2, points_3, ...]
    # patch_1 : numpy array of shape(patch_size,patch_size,3)
    # point_1 : numpy array of shape (N,2): N number of TILs.
    # patch_size = size of the patch images.
    
    # outputs: list of augmneted patches and points. Increased 4 times.
    aug_patches = []
    aug_patches_points = []    
    for i in range(len(patches)):        
        aug_patches.append(patches[i])
        aug_patches_points.append(patches_points[i])
            
    for i in range(len(patches)):              
        aug_patches.append(np.fliplr(patches[i])) 
        if patches_points[i].shape[0]>0:
            temp = np.copy(patches_points[i])
            temp[:,0]= patch_size-temp[:,0]
            aug_patches_points.append(temp)  
        else:
            aug_patches_points.append(np.array([]))  
        
    for i in range(len(patches)):     
        aug_patches.append(np.flipud(patches[i]))
        if patches_points[i].shape[0]>0:
            temp = np.copy(patches_points[i])
            temp[:,1]= patch_size-temp[:,1]
            aug_patches_points.append(temp) 
        else:
            aug_patches_points.append(np.array([]))        

    for i in range(len(patches)):  
        aug_patches.append(np.flipud(np.fliplr(patches[i])))
        if patches_points[i].shape[0]>0:
            temp = np.copy(patches_points[i])
            temp[:,0]= patch_size-temp[:,0]
            temp[:,1]= patch_size-temp[:,1]
            aug_patches_points.append(temp) 
        else:
            aug_patches_points.append(np.array([]))        
    return aug_patches, aug_patches_points
