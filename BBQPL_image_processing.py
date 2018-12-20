#!/usr/bin/env python3

import numpy as np
import os,sys
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage import data, exposure, img_as_float


# Helper functions

def load_image(infilename):
    data = mpimg.imread(infilename)
    return data
###############################################################
def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

###############################################################

def improve_image_contrast(imgs_array):
    """
    Description : 
    ---------------
    Improves the contrast of an image pased on pratical experience
    
    Parameters:
    ---------------
    imgs_array: np.array
        3D or 4D numpy array. is an array containing images in grey scale or RGB.
    
    Returns : 
    ---------------
    imgs_array: np.array
        numpy array of the same size of the input with augmented contrast.
    """
    
    
    for img in range(imgs_array.shape[0]):
        v_min, v_max = np.percentile(imgs_array[img], (3.0, 97.0))
        imgs_array[img] = exposure.rescale_intensity(imgs_array[img], in_range=(v_min, v_max))
    return imgs_array
#############################################################

def contineous_to_binary_mask(mask, threshold = 0.25):
    """
    Description :
    ---------------
    Transform a mask with contineous values into a binary mask given the threshold.
    
    Parameters:
    ---------------
    mask: np.array
        2D numpy array with the contineous mask
    
    Returns : 
    --------------
    mask : np.array
        2D numpy array with the binary mask
    """
    
    mask[np.where(mask>threshold)] = 1
    mask[np.where(mask<=threshold)] = 0
    
    return mask

#############################################################


def value_to_class(v, foreground_threshold):
    df = np.sum(v)
    if df > foreground_threshold:
        return 1
    else:
        return 0
    
###########################################################

def imgs_array_to_imgs_patch_array(images_array, patch_size = 128, pixel_increment = 16):
    """
    Description : 
    -----------------
    Transform an array of grey-scale images into an array of overlapping grey scale patches.
    
    Parameters : 
    -----------------
    imgs_array : np.array
        3D numpy array containing grey scale images.
    patch_size : int.
        Assuming a square patches, this is the length of the side of the patch in pixels .
    pixel_increments = 16 : int.
        number of pixels to shift one patch from the next.
        
    Returns :
    ---------------
    imgs_patchs_array : np.array.
        3D dimenssionnal numpy array containing the patches.
    """
    
    nb_pose = int((images_array.shape[1]-patch_size)/pixel_increment) + 1

    imgs_patchs_array = np.zeros((int(images_array.shape[0] * nb_pose**2), patch_size, patch_size))
   
    print("Convert images to patches, {0} -> {1}, with {2} patches per images...".format(images_array.shape, 
                                                                                         imgs_patchs_array.shape, 
                                                                                         nb_pose**2))
    
    for i in range(images_array.shape[0]):
        for row in range(nb_pose):
            for col in range(nb_pose):
                imgs_patchs_array[i*nb_pose**2 + nb_pose*row+col, :, :] = images_array[i, row * pixel_increment : row * pixel_increment 
                                                                                       + patch_size, col * pixel_increment : 
                                                                                        col * pixel_increment + patch_size ]
               

    return imgs_patchs_array

#################################################################

def imgs_patch_array_to_imgs_array(imgs_patchs_array, nb_imgs, image_size, pixel_increment = 16):
    """
    Description : 
    -----------------
    Transform an array of grey-scale images into an array of overlapping grey scale patches.
    
    Parameters : 
    -----------------
    imgs_patchs_array : np.array.
        3D dimenssionnal numpy array containing the patches.
    nb_imgs : int.
        Number of images to recover 
    image_size : int.
        Assuming a square image, this is the length of the side of the image in pixels .
    pixel_increments = 16 : int.
        number of pixels to shift one patch from the next.
        
    Returns :
    ---------------
    imgs_array : np.array
        3D numpy array containing grey scale images.
    """
    
    patch_size = imgs_patchs_array.shape[1]
    imgs_array = np.zeros((nb_imgs, image_size, image_size))
    mask_to_average = np.zeros(imgs_array.shape)
    nb_pose = int((imgs_array.shape[1] - patch_size) / pixel_increment) + 1
    nb_pose_per_image = nb_pose ** 2
    
    print("Convert patches to images, {0} -> {1}, with {2} patches per images...".format(imgs_patchs_array.shape, 
                                                                                         imgs_array.shape, 
                                                                                         nb_pose_per_image))
                          
    for i in range(0, imgs_patchs_array.shape[0], nb_pose_per_image):
        
        img_id = int(i/nb_pose_per_image)
        
        for k, img_patch in enumerate(imgs_patchs_array[img_id * nb_pose_per_image: (img_id + 1) * nb_pose_per_image]):
            
            row = int(k//nb_pose)
            col = int(k%nb_pose)
            imgs_array[img_id, row * pixel_increment : row * pixel_increment + patch_size, col * pixel_increment : 
                      col* pixel_increment + patch_size ] += img_patch
            mask_to_average[img_id, row * pixel_increment : row * pixel_increment + patch_size, 
                            col * pixel_increment : col * pixel_increment + patch_size] += np.ones(img_patch.shape)

    imgs_array = imgs_array/np.float32(mask_to_average)
 

    return imgs_array
    
    
