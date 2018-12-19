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
                 u_net_suffix = '',
                 training_images_dir = 'training/images/', 
                 training_masks_dir = 'training/groundtruth/', 
                 testing_images_dir = 'test_set_images/', 
                 predicted_dir = 'results/',
                 submission_filename = 'submission',
                 training_dataset_length = 'all',
                 testing_dataset_length = 'all',
                 training_strides = 16,
                 testing_strides = 16,
                 patch_size = 16,
                 batch_size = 16,
                 epochs = 5,
                 nb_filters = 16,
                 test_ratio = 0.2, 
                 validation_ratio = 0.2, 
                 preprocess = True,
                 seed = 1, 
                 vertical_flip = False, 
                 horizontal_flip = False, 
                 random_rotation = 0,
                 make_training = True,
                 make_prediction = True):
        """
        Descrition :
        -------------
            Creates a new instance of the class Pipleline. This class allows the creation and the training of a Unet model based on a training set and several parameters to design it. This class also allow to load an existing model and load a set of images to test its predictions. 
        
        
        Parameters :
        -------------
            u_net_file = None, string.
                Name of the existing Unet model to load.
            u_net_suffix = '', string.
                Suffix to add at the end of the model file when saving it.
            training_images_dir = 'training/images/', string.
                relative path to the directory that contains the training data set images.
            training_masks_dir = 'training/groundtruth/', string.
                relative path to the directory that contains the training data set masks.
            testing_images_dir = 'test_set_images/', string.
                relative path to the directory that contains the testing data set images.
            predicted_dir = 'results/', string.
                relative path to the directory that will contain the predicted masks.
            submission_filename = 'submission', string.
                Name of the file to submit to the CrowdAI plateform. The file will be saved as a '.csv'.
            training_dataset_length = 'all', int.
                Length of the trainig dataset to use. if 'all', take the full dataset.
            testing_dataset_length = 'all', int.
                Length of the testing dataset to use. if 'all', take the full dataset.
            training_strides = 16, int.
                Length of the strides to use on the training set to move the patch on the image. 
            testing_strides = 16, int.
                Length of the strides to use on the testing set to move the patch on the image. 
            patch_size = 16, int.
                Size of the square patches that are used to cut the image and feed the model. 
            batch_size = 16, int.
                Batch size for the SGD training alogithm.
            epochs = 5, int.
                Number of epochs for the training.
            nb_filters = 16, int.
                Number of filters the unet model should start with. 
            test_ratio = 0.2, float. 
                Ratio of the training set to use for testing, at the end of the training. (test_ratio < 1) 
            validation_ratio = 0.2, float.
                Ratio of the training set to use for validation, at the end of each epoch. (validation_ratio < 1) 
            preprocess = True, bool.
                Apply image preprocessing before training.
            seed = 1, int. 
                Defines random constant.
            vertical_flip = False, bool
                Apply vertical flip to the training set before training for data augmentation.
            horizontal_flip = False, bool.
                Apply horizontal flip to the training set before training for data augmentation.
            random_rotation = 0, int.
                Number of random rotations per image to apply to the training set before training for data augmentation.
            make_training = True, bool.
                Perform training if True.
            make_prediction = True
                Perform prediction if True.
        
        Returns :
        -----------
        Pipeline instance.

        """
        
        
        
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
                                    preprocess = preprocess,
                                    random_rotation = random_rotation,
                                    u_net_suffix = u_net_suffix,
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
                                    preprocess = preprocess,
                                    predicted_dir = self.predicted_dir,
                                    submission_filename = self.submission_filename,
                                    seed = self.seed)
            self.predicting.process_data()
            self.predicting.predict_masks()
            self.predicting.save_prediction()
            self.predicting.create_submission_file()
        

        
        
