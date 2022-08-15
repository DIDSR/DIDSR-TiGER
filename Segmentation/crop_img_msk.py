import numpy as np

def crop_img_msk(img, msk, size, stride):
    # this function crops an image and its corresponding mask using the sliding window technique. Size is the patch size of the sliding window and stride is the stride of the sliding window patches.
    cropped_img = []  
    cropped_msk = []     
    size_x = img.shape[1]
    size_y = img.shape[0]
    x = np.arange(0, size_x, stride)
    y = np.arange(0, size_y, stride)
    x = [i for i in x if i+size <= size_x]
    y = [i for i in y if i+size <= size_y]
    x.append(size_x-size)
    y.append(size_y-size)
    x = np.unique(x)
    y = np.unique(y)
    for i in x:
        for j in y:
            cropped_img.append(img[j:j+size, i:i+size])
            cropped_msk.append(msk[j:j+size, i:i+size])
    return cropped_img, cropped_msk
