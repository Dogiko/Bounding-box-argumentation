import cv2
import numpy as np
import torch
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
    extend = 4*max(half_weith, half_hight)
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

def random_crop_square_full_box(img, start, end, n_crop=1, resize=-1, box_in_crop_ratio=2., flip=False):
    cheak_type(resize, int)
    cheak_type(n_crop, int)
    cheak_interval(box_in_crop_ratio, mini=1)
    
    output = []
    box_center = np.array([start[0]+end[0], start[1]+end[1]])/2
    half_weith = int((end[0]-start[0]+1)/2)
    half_hight = int((end[1]-start[1]+1)/2)
    extend = int(2*box_in_crop_ratio*max(half_weith, half_hight))+1
    img = ext_black(img, extend)
    box_center += extend
    crop_ratio = np.random.rand(n_crop)*(box_in_crop_ratio-1)+1
    crop_half_length = crop_ratio*max(half_weith, half_hight)
    for c in range(n_crop):
        crop_center = box_center + (2*np.random.rand(2)-1)*max(half_weith, half_hight)*(crop_ratio[c]-1)
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

def random_crop_square_nothing(img, starts, ends, n_crop=1, resize=-1, avoid_ratio=[0.707, 1.414], flip=False, minimum_length=32, max_try=100):
    cheak_type(resize, int)
    cheak_type(n_crop, int)
    cheak_interval(avoid_ratio[1], mini=0)
    cheak_interval(avoid_ratio[0], mini=0, maxi=avoid_ratio[1], left_close=True, right_close=True)
    cheak_interval(minimum_length, mini=1, maxi=np.min(img.shape[:2]), left_close=True,
                   message="minimum_length error, expect int between 1 and image_size(smaller side) got {}".format(minimum_length))
    
    output = []
    for c in range(n_crop):
        for t in range(max_try):
            length = np.random.randint(minimum_length, np.min(img.shape[:2])+1)
            y = np.random.randint(img.shape[0]+1-length)
            x = np.random.randint(img.shape[1]+1-length)

            centers = (starts + ends)/2
            center_in_square_bool = (centers[:,0]>=x)*(centers[:,0]<x+length)*(centers[:,1]>=y)*(centers[:,1]<y+length)
            box_in_square_ratio = (ends - starts)[center_in_square_bool].max(axis=1)/length
            if not ((box_in_square_ratio>=avoid_ratio[0])*(box_in_square_ratio<=avoid_ratio[1])).any():
                
                crop_img = crop_square(img, (x, y), length)
                
                if resize>0:
                    crop_img = cv2.resize(crop_img, (resize, resize)) 
                
                if flip:
                    if np.random.rand()>0.5:
                        crop_img = cv2.flip(crop_img, 1)
                
                output.append(crop_img)
                break
    
    return output

def img_to_torch(numpy_img):
    return torch.tensor(np.transpose(numpy_img.astype(np.float32), (0,3,1,2))/127.5 - 1)