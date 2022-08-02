import cv2
import math
from crop_img_msk import crop_img_msk
import numpy as np

def rotation(image, angleInDegrees):
    h, w = image.shape[:2]
    img_c = (w / 2, h / 2)

    rot = cv2.getRotationMatrix2D(img_c, angleInDegrees, 1)

    rad = math.radians(angleInDegrees)
    sin = math.sin(rad)
    cos = math.cos(rad)
    b_w = int((h * abs(sin)) + (w * abs(cos)))
    b_h = int((h * abs(cos)) + (w * abs(sin)))

    rot[0, 2] += ((b_w / 2) - img_c[0])
    rot[1, 2] += ((b_h / 2) - img_c[1])

    outImg = cv2.warpAffine(image, rot, (b_w, b_h), flags=cv2.INTER_LINEAR)
    return outImg

def extract_patches_with_angle(img, msk, size, stride, angle):

    img_rotated = rotation(img, angle)
    msk_rotated = rotation(msk, angle)

    imgs, msks = crop_img_msk(img_rotated, msk_rotated, size, stride)

    idx = [i for i in range(len(msks)) if np.any(msks[i]==0)]   

    imgs_wanted = []
    msks_wanted = []
    for i in range(len(imgs)):
        if i not in idx:
            imgs_wanted.append(imgs[i])
            msks_wanted.append(msks[i])
            
    return imgs_wanted, msks_wanted
        
        

