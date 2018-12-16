#!/usr/bin/env python3

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from BBQPL_ProcessDataSet import *
from BBQPL_Unet import *
from BBQPL_image_processing import *
from mask_to_submission import masks_to_submission
import scipy.misc
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os,sys


class Predicting():
    
    def __init__(self, images_dir = None,
                 predicted_dir = 'results/',
                 submission_filename = 'submission',
                 u_net = None,
                 dataset_length = 'all',
                 patch_size = 128, 
                 strides = 16,
                 seed = 1):
        
        self.images_dir = images_dir
        self.predicted_dir = predicted_dir
        self.submission_filename = submission_filename
        self.u_net = u_net
        self.dataset_length = dataset_length
        self.patch_size = patch_size
        self.strides = strides
        self.seed = seed
        
        self.test_ratio = 1
        
        assert self.u_net is not None, "You must give a model to the \'Predicting()\' class."
        
        self.X = LoadTestSet(imgs_path = images_dir, dataset_length = dataset_length, datatype = 'images', 
                                patch_size = patch_size, strides = strides, test_ratio = self.test_ratio, seed = seed)
        
        self.image_shape = (self.X.imgs_array.shape[1], self.X.imgs_array.shape[2])
        
        
    def process_data(self):            
        self.X.process_data()
        
    def predict_masks(self):
        
        assert self.X.imgs_test is not None and self.X.imgs_test.shape[3] == 1, "Data must be processed before it model can predict it."
        print('Predicting masks from test set ...')
        Z_patch = self.u_net.predict(self.X.imgs_test)
        self.resize_patches_to_images(patch = Z_patch)    
    
    def resize_patches_to_images(self, patch = None, nb_imgs= None, imgs_side_size = None, patch_size= None):
        
        if patch is None: patch = self.X.imgs_patch
        if nb_imgs is None: nb_imgs = self.dataset_length
        if imgs_side_size is None: imgs_side_size = self.X.imgs_augm.shape[1]
        
        
        self.masks_array = imgs_patch_array_to_imgs_array(np.squeeze(patch), 
                                                     nb_imgs = nb_imgs, 
                                                     image_size = imgs_side_size, 
                                                     pixel_increment = self.strides)
        
    
    def save_prediction(self, predicted_dir = None):
        if predicted_dir is None: predicted_dir = self.predicted_dir
            
        print("Saving predictions to folder : {0} ...".format(predicted_dir))
        if not os.path.exists(predicted_dir):
            os.makedirs(predicted_dir)
        for i, img in enumerate(self.masks_array):
            scipy.misc.toimage(img, cmin=0.0, cmax=1.0).save(predicted_dir + self.X.repositories[i] + '.png')
    
    def create_submission_file(self, submission_filename = None, predicted_dir = None):
        
        if predicted_dir is None: predicted_dir = self.predicted_dir
        if submission_filename is None: submission_filename = self.submission_filename
        
        print("Creating submission file : {0}.csv ...".format(submission_filename))
        if not os.path.exists(predicted_dir):
            os.makedirs(predicted_dir)
        
        files = os.listdir(predicted_dir)
        tot_files = [predicted_dir + x for x in files]
        
        masks_to_submission(submission_filename + '.csv', *tot_files)
        
    def display_image_prediction(self, img_obj = None, mask_obj = None, range_ = None, mod = None):

        if img_obj is None: img_obj = self.X.imgs_array
        if mask_obj is None: mask_obj = self.masks_array
        if range_ is None: range_ = range(img_obj.shape[0])
        if mod is None: mod = int(img_obj.shape[0]/10)

        mod = max(4, mod)
        for img, i  in enumerate(range_):
            if i%mod == 0:
                plt.figure(figsize = (10, 4))
                plt.subplot(1,4,1)
                plt.imshow(np.squeeze(img_obj[i]), cmap='Greys_r')
                plt.title("Image {0}".format(i))
                plt.subplot(1,4,2)
                plt.imshow(np.squeeze(mask_obj[i]), cmap='Greys_r')
                plt.subplot(1,4,3)
                plt.imshow(np.squeeze(img_obj[i+int(mod/2)]), cmap='Greys_r')
                plt.subplot(1,4,4)
                plt.imshow(np.squeeze(mask_obj[i+int(mod/2)]), cmap='Greys_r')
                plt.show()
                
        