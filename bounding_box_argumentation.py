import cv2
import numpy as np
from cheak_tools import cheak_type, cheak_interval

def ext_black(img, extend):
    cheak_type(extend, int)
    cheak_interval(extend, mini=0, left_close=True)
    output = np.zeros((img.shape[0] + 2*extend, img.shape[1] + 2*extend, img.shape[2]), np.uint8)
    output[extend:-extend, extend:-extend] = img
    return output

def crop_square(img, start, length):
    cheak_type(length, int)
    cheak_interval(length, mini=0, left_close=False)
    output = img[start[1]:start[1]+length, start[0]:start[0]+length]
    # note order of x y in cv2-array is array[y, x]
    return output

def random_crop_square(img, start, end, n_crop=1, resize=-1, box_in_crop_ratio=[0.707, 1.414], flip=False):
    cheak_type(resize, int)
    cheak_type(n_crop, int)
    cheak_interval(box_in_crop_ratio[1], mini=0)
    cheak_interval(box_in_crop_ratio[0], mini=0, maxi=box_in_crop_ratio[1], left_close=True, right_close=True)
    
    output = []
    box_center = np.array([start[0]+end[0], start[1]+end[1]])/2
    half_weith = int((end[0]-start[0]+1)/2)
    half_hight = int((end[1]-start[1]+1)/2)
    extend = 2*max(half_weith, half_hight)
    img = ext_black(img, extend)
    box_center += extend
    crop_ratio = np.random.rand(n_crop)*(box_in_crop_ratio[1]-box_in_crop_ratio[0])+box_in_crop_ratio[0]
    crop_half_length = crop_ratio*max(half_weith, half_hight)
    for c in range(n_crop):
        crop_center = box_center + (np.random.rand(2)*2*crop_half_length[c] - crop_half_length[c])
        crop_img = (crop_square(img,
                                (crop_center-crop_half_length[c]).astype(np.int),
                                int(2*crop_half_length[c])
                               )
                   )
        if resize>0:
            crop_img = cv2.resize(crop_img, (resize, resize)) 
        
        crop_box_center = box_center - crop_center + crop_half_length[c]
        crop_start = crop_box_center - np.array([half_weith, half_hight])
        crop_end = crop_box_center + np.array([half_weith, half_hight])
        crop_box_center /= 2*crop_half_length[c]
        crop_half_weith = half_weith/(2*crop_half_length[c])
        crop_half_hight = half_hight/(2*crop_half_length[c])
        
        if flip:
            if np.random.rand()>0.5:
                crop_img = cv2.flip(crop_img, 1)
                crop_box_center[0] = 1- crop_box_center[0]
        
        output.append([crop_img, crop_box_center, crop_half_weith, crop_half_hight])
    
    return output