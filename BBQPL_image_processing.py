#!/usr/bin/env python3

import numpy as np
import os,sys
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from tqdm import tqdm, tnrange
from skimage import data, exposure, img_as_float




# Helper functions

def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def img_crop(im, w, h):
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

#######################################################

def label_to_img(imgwidth, imgheight, w, h, labels):
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            im[j:j+w, i:i+h] = labels[idx]
            idx = idx + 1
    return im

def label_patch_to_img_patch(imgwidth, imgheight, w, h, labels):
    im = np.zeros([imgwidth, imgheight])
    print("Creating mask image of size ({0}, {1}), using patches of size ({2}, {3}) ...".format(imgwidth, imgheight, w, h))
    
    idx_i = 0
    for i in range(0,imgheight,h):
        idx_j = 0
        for j in range(0,imgwidth,w):
            #print("idx_j = {0}".format(idx_j))
            im[j:j+w, i:i+h] = labels[idx_j, idx_i]
            idx_j = idx_j + 1
        idx_i = idx_i + 1
    return im

def label_to_img_patch(imgwidth, imgheight, w, h, labels):
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            im[j:j+w, i:i+h] = labels[idx, :, :]
            idx = idx + 1
    return im

def patch_to_img(patches):
    
    imgheight = 400
    imwidth = 400
    im = np.zeros([imwidth,imgheight])
    h = 16
    w = 16
    idx=0
    for i in range(0,imgheight,h):
        for j in range(0,imwidth,w):
            im[j:j+w, i:i+h] = patches[idx, :, :]
            idx = idx + 1
    return im

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

############################################################

def improve_image_contrast(imgs_array):
    
    for img in range(imgs_array.shape[0]):
        v_min, v_max = np.percentile(imgs_array[img], (3.0, 97.0))
        imgs_array[img] = exposure.rescale_intensity(imgs_array[img], in_range=(v_min, v_max))
    return imgs_array
#############################################################

def contineous_to_binary_mask(mask, threshold = 0.25):
    
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
    
############################################################


def resize_image_array(array_of_images, new_size = (512, 512)):
    
    new_array_of_image = np.zeros((array_of_images.shape[0], new_size[0], new_size[1]))
    
    for image in range(array_of_images.shape[0]):
            new_array_of_image[image,:,:] = cv.resize(array_of_images[image,:,:], new_size)
            
    return new_array_of_image

###########################################################

def imgs_array_to_imgs_patch_array(images_array, patch_size = 128, pixel_increment = 16):
    
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


def imgs_patch_array_to_imgs_array(imgs_patchs_array, nb_imgs, image_size, pixel_increment = 16):
    
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
    
    
