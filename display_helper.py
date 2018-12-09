import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os,sys
from PIL import Image
import numpy as np
import cv2 as cv
from image_basic_operations import value_to_class, label_to_img, make_img_overlay



def display_greyimg_mask_pred(grey_image, mask, pred, display = True):
    
    true_img = np.concatenate((grey_image, mask), axis = 1)
    true_img = cv.resize(true_img, (int(true_img.shape[0]), int(true_img.shape[1]/4)))
    print(true_img.shape, pred.shape)
    full_img = np.concatenate((pred, true_img), axis = 0)
    
    plt.imshow(full_img, cmap='Greys_r')
    plt.show
    
    return full_img
    
    
def img_crop2(im, w, h):
    
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches


def display_img_with_pred(img, pred, display = True):
    
    patch_size = 16
    
    pred_patches = []
    imgwidth = img.shape[0]
    imgheight = img.shape[1]
    is_2d = len(img.shape) < 3
    for i in range(0,imgheight,patch_size):
        for j in range(0,imgwidth,patch_size):
            if is_2d:
                im_patch = img[j:j+patch_size, i:i+patch_size]
            else:
                im_patch = img[j:j+patch_size, i:i+patch_size, :]
            pred_patches.append(im_patch)
    
    #pred_patches = im_crop2(mask, patch_size, patch_size)
    labels = np.asarray([value_to_class(np.mean(pred_patches[i]), 0.25) for i in range(len(pred_patches))])
    
    w = pred.shape[0]
    h = pred.shape[1]
    
    pred_img = label_to_img(w, h, patch_size, patch_size, labels) 
    overlay_img = make_img_overlay(img, pred_img)
    
    if display :
        plt.imshow(overlay_img)
        plt.show
    
    return overlay_img
    
    
