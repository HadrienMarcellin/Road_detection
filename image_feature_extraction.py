import numpy as np
import os, sys
from skimage import data, exposure, img_as_float


# Extract 6-dimensional features consisting of average RGB color as well as variance
def extract_features(img):
    feat_m = np.mean(img, axis=(0,1))
    feat_v = np.var(img, axis=(0,1))
    feat = np.append(feat_m, feat_v)
    return feat

# Extract 2-dimensional features consisting of average gray color as well as variance
def extract_features_2d(img):
    feat_m = np.mean(img)
    feat_v = np.var(img)
    feat = np.append(feat_m, feat_v)
    return feat

# Extract features for a given image
def extract_img_features(filename):
    img = load_image(filename)
    img_patches = img_crop(img, patch_size, patch_size)
    X = np.asarray([ extract_features(img_patches[i]) for i in range(len(img_patches))])
    return X

def improve_image_contrast(imgs_array):
    
    for img in range(imgs_array.shape[0]):
        v_min, v_max = np.percentile(imgs_array[img], (3.0, 97.0))
        imgs_array[img] = exposure.rescale_intensity(imgs_array[img], in_range=(v_min, v_max))
    return imgs_array