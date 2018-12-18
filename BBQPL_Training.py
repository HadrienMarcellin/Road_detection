#!/usr/bin/env python3

import numpy as np
import numpy.random
from BBQPL_compute_f1 import Metrics
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os,sys
from BBQPL_image_processing import *
from BBQPL_ProcessDataSet import *
from BBQPL_Unet import *


           
                
class Training:
    
    def __init__(self, imgs_path = None, 
                 masks_path = None, 
                 dataset_length = 'all',
                 test_ratio = None, 
                 validation_ratio = None, 
                 patch_size = 128,
                 strides = 16, 
                 batch_size = 16,
                 epochs = 5,
                 nb_filters = 16,
                 vertical_flip = False, 
                 horizontal_flip = False, 
                 preprocess = True,
                 random_rotation = 0,
                 u_net_suffix = '',
                 seed = 1):
        
       

        self.X = LoadTrainSet(imgs_path = imgs_path, dataset_length = dataset_length, datatype = 'images', 
                              patch_size = patch_size, strides = strides, test_ratio = test_ratio, 
                              vertical_flip = vertical_flip, horizontal_flip = horizontal_flip, 
                              preprocess = preprocess, random_rotation = random_rotation, seed = seed)
        self.Y = LoadTrainSet(imgs_path = masks_path, dataset_length = dataset_length, datatype = 'masks', 
                              patch_size = patch_size, strides = strides, test_ratio = test_ratio, 
                              vertical_flip = vertical_flip, horizontal_flip = horizontal_flip, 
                              preprocess = preprocess, random_rotation = random_rotation, seed = seed)
        
        self.dataset_length = dataset_length
        self.epochs = epochs
        self.test_ratio = test_ratio
        self.validation_ratio = validation_ratio
        self.batch_size = batch_size
        self.metrics = Metrics()
        self.nb_filters = nb_filters
        self.patch_size = patch_size
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.random_rotation = random_rotation
        
        self.u_net = UnetModel(nb_filters = self.nb_filters,
                                patch_size = self.patch_size, 
                                horizontal_flip = self.horizontal_flip, 
                                vertical_flip = self.vertical_flip, 
                                random_rotation = self.random_rotation, 
                                u_net_suffix = u_net_suffix)
        
        
        
    def process_data(self, test_ratio = None):
        if test_ratio is None: test_ratio = self.test_ratio
            
        self.X.process_data(test_ratio = test_ratio)
        self.Y.process_data(test_ratio = test_ratio)
    
    def fit_model(self, validation_ratio = None, epochs = None, batch_size = None):
        
        if validation_ratio is None: validation_ratio = (self.validation_ratio or 0)
        if epochs is None: epochs = (self.epochs or 5)
        if batch_size is None: batch_size = (self.batch_size or 16)
            
        assert self.X.imgs_train is not None and self.Y.imgs_train is not None, "Data must be preprocessed before the model is feed."
        
        self.u_net.build_model()
        self.u_net.compile_model()
        self.u_net.train(X_input = self.X.imgs_train, 
                                        Y_input = self.Y.imgs_train, 
                                        validation_ratio=validation_ratio, 
                                        epochs=epochs,
                                        batch_size=batch_size, 
                                        callbacks=[self.metrics])
    
    def compute_error_loss():
        return error, loss
    
    def display_image_mask(self, img_obj = None, mask_obj = None, range_ = None, mod = None):
        
        if img_obj is None: img_obj = self.X.imgs_patch
        if mask_obj is None: mask_obj = self.Y.imgs_patch
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
    
    def display_range_images():
        return figure
    
    def dispaly_random_images():
        return figure
