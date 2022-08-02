import numpy as np

def aug_imgs_points(patches, patches_points, patch_size = 128):
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