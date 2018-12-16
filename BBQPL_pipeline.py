#!/usr/bin/env python3

import numpy as np
import numpy.random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os,sys
from BBQPL_ProcessDataSet import *
from BBQPL_Training import *
from BBQPL_Predicting import *
from BBQPL_Unet import *


class Pipeline:
    
    def __init__(self, 
                 u_net_file = None, 
                 training_images_dir = None, 
                 training_masks_dir = None, 
                 testing_images_dir = None, 
                 predicted_dir = 'results/',
                 submission_filename = 'submission',
                 training_dataset_length = 'all',
                 testing_dataset_length = 'all',
                 training_strides = 16,
                 testing_strides = 16,
                 patch_size = 128,
                 batch_size = 16,
                 epochs = 5,
                 nb_filters = 16,
                 test_ratio = 0.2, 
                 validation_ratio = 0.2, 
                 seed = 1, 
                 vertical_flip = False, 
                 horizontal_flip = False, 
                 random_rotation = 0,
                 make_training = True,
                 make_prediction = True):
        
        print("Initiating pipeline ...")
        
        self.u_net_file = u_net_file
        
        self.training_images_dir = training_images_dir
        self.training_masks_dir = training_masks_dir
        self.testing_images_dir = testing_images_dir
        self.predicted_dir = predicted_dir
        self.submission_filename = submission_filename
        
        self.training_dataset_length = training_dataset_length
        self.testing_dataset_length = testing_dataset_length
        self.training_strides = training_strides
        self.testing_strides = testing_strides
        
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.nb_filters = nb_filters
        self.test_ratio = test_ratio
        self.validation_ratio = validation_ratio
        self.seed = seed
        
        if self.u_net_file is None and make_training is True: 
            self.training = Training(imgs_path = self.training_images_dir, 
                                    masks_path = self.training_masks_dir, 
                                    dataset_length = self.training_dataset_length,
                                    strides = self.training_strides, 
                                    patch_size = self.patch_size, 
                                    batch_size = self.batch_size, 
                                    test_ratio = self.test_ratio, 
                                    validation_ratio = self.validation_ratio, 
                                    epochs = self.epochs,
                                    nb_filters = self.nb_filters,
                                    vertical_flip = vertical_flip,
                                    horizontal_flip = horizontal_flip, 
                                    random_rotation = random_rotation,
                                    seed = self.seed)
            
            self.training.process_data()
            self.training.fit_model()
            self.u_net = self.training.u_net
        
        else:
            self.u_net = UnetModel()
            self.u_net.load_model(u_net_file)
        
        if make_prediction is True:
            self.predicting = Predicting(images_dir = self.testing_images_dir, 
                                    u_net = self.u_net,
                                    dataset_length = self.testing_dataset_length,
                                    patch_size = self.patch_size, 
                                    strides = self.testing_strides, 
                                    predicted_dir = self.predicted_dir,
                                    submission_filename = self.submission_filename,
                                    seed = self.seed)
            self.predicting.process_data()
            self.predicting.predict_masks()
            self.predicting.save_prediction()
            self.predicting.create_submission_file()
        
                

    
class graphic_display:
    
    def __init__(self):
        return
    
    def display_image_mask(self, image1, image2):
        
        assert image1.shape[1] == image2.shape[1]
        
        cimg = np.concatenate((image1, image2), axis = 1)
        plt.imshow(cimg, cmap='Greys_r')
        plt.show
        return cimg
    
    def display_range_images(self):
        return figure
    
    def dispaly_random_images(self):
        return figure
        
        
