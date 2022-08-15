import numpy as np

def aug_imgs_msks_d4(patches, patches_masks):
    # This function augments patches and masks using D4 symmetry of a square.
    # Inputs: patches and patches_masks are lists containing patches and patches_masks.
    # Outputs: aug_patches and aug_patches_masks: are lists containing the augmented patches and masks using D4 symmetry of a sqaure.
    
    aug_patches = []
    aug_patches_masks = []
    for i in range(len(patches)):        
        aug_patches.append(patches[i])
        aug_patches_masks.append(patches_masks[i])
            
    for i in range(len(patches)):     
        aug_patches.append(np.fliplr(patches[i]))
        aug_patches_masks.append(np.fliplr(patches_masks[i]))            
        
    for i in range(len(patches)):    
        aug_patches.append(np.flipud(patches[i]))
        aug_patches_masks.append(np.flipud(patches_masks[i]))                

    for i in range(len(patches)):    
        aug_patches.append(np.rot90(patches[i]))
        aug_patches_masks.append(np.rot90(patches_masks[i]))   
        
    for i in range(len(patches)):    
        aug_patches.append(np.rot90(np.rot90(patches[i])))
        aug_patches_masks.append(np.rot90(np.rot90(patches_masks[i])))
        
    for i in range(len(patches)):    
        aug_patches.append(np.rot90(np.rot90(np.rot90(patches[i]))))
        aug_patches_masks.append(np.rot90(np.rot90(np.rot90(patches_masks[i]))))
    
    for i in range(len(patches)):    
        aug_patches.append(np.fliplr(np.rot90(patches[i])))
        aug_patches_masks.append(np.fliplr(np.rot90(patches_masks[i])))
        
    for i in range(len(patches)):    
        aug_patches.append(np.flipud(np.rot90(patches[i])))
        aug_patches_masks.append(np.flipud(np.rot90(patches_masks[i])))        

    return aug_patches, aug_patches_masks
