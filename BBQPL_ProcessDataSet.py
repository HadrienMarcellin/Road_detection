#!/usr/bin/env python3

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os,sys
import numpy as np
import numpy.random
from scipy.ndimage import rotate
from BBQPL_image_processing import *






class ProcessDataSet:
        
    def __init__(self, imgs_path = None,
                 imgs_array = None,
                 dataset_length = None,
                 patch_size = 128, 
                 strides = 16, 
                 batch_size = 16, 
                 datatype = None, 
                 test_ratio = 0, 
                 vertical_flip = False, 
                 horizontal_flip = False, 
                 random_rotation = 0,
                 preprocess = True,
                 seed = 1):
        
        """
        Descrition :
        -------------
            Creates a new instance of the class ProcessDataSet. This class allows the loading and processing of a unique set of image. This class also allow to load an existing array of image and performs image preprocessing, data augmentation, format conversion from images to patches and data spliting according to the test ratio. 
        
        
        Parameters :
        -------------
            
            imgs_path = None, string.
                relative path to the directory that contains the training data set images.
            imgs_array = None, np.array.
                Array of images. if not None, overrides imgs_path.
            dataset_length = 'all', int.
                Length of the trainig dataset to use. if 'all', take the full dataset.
            strides = 16, int.
                Length of the strides to use on the training set to move the patch on the image. 
            patch_size = 16, int.
                Size of the square patches that are used to cut the image and feed the model. 
            batch_size = 16, int.
                Batch size for the SGD training alogithm. (deprecated, not used)
            test_ratio = 0.2, float. 
                Ratio of the training set to use for testing, at the end of the training. (test_ratio < 1) 
            datatype = 'None', string.
                'images' or 'masks'.
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
            
        
        Returns :
        -----------
        ProcessDataSet instance.

        """
        
        print("Initiating dataset : {0} ...".format(datatype))
        
        self.imgs_path = imgs_path
        self.dataset_length = dataset_length
        self.patch_size = patch_size
        self.strides = strides
        self.batch_size = batch_size
        self.datatype = datatype
        self.test_ratio = test_ratio
        self.preprocess = preprocess
        
        self.vertical_flip = vertical_flip 
        self.horizontal_flip = horizontal_flip
        self.random_rotation = random_rotation
     
        self.imgs_train = None
        self.imgs_test = None
        
        if imgs_array is None: 
            self.load_images() 
        else:
            self.imgs_array = imgs_array
            self.dataset_length = self.imgs_array.shape[0]
        
        


    
    def image_preprocessing(self, X = None, pixel_threshold = 0.25):
        
        """
        Description:
        --------------
        Perform image preprocessing depending on the datatype. Improves contrast for 'images' type and apply a binary filter on 'masks' type.
        
        Parameters:
        --------------
        X = None, np.array
            array of images to process
        pixel_threshold = 0.25, float
            thresholf for binary filter for 'masks'
        
        """
        
        
        if X is None: X = self.imgs_array
                    
        if len(X.shape) == 4 and self.datatype == 'images':
            print("Processing {0}... Improve contrast and performing grey coloration.".format(self.datatype))
            if self.preprocess:
                X = improve_image_contrast(X)
            self.imgs_array = np.mean(X, axis = 3)
        if self.datatype == 'masks':
            print("Processing {0}... Make binary mask with threshold : {1}.".format(self.datatype, pixel_threshold))
            self.imgs_array = contineous_to_binary_mask(X, pixel_threshold)
     
        assert (len(self.imgs_array.shape) > 2), "Input must be an array of images of size \
                                                    (N*d*d) for grey scales images or (N*d*d*3) \
                                                    for color images. Actual size is {0}".format(X.shape)
    
    
    
        
        
    def resize_dataset_to_fit_model(self, X = None, patch_size = None, strides = None):
        """
        Description:
        --------------
        Converts image array to patch array.
        
        Parameters:
        --------------
        X = None, np.array
            array of images to process
        strides = None, int.
                Length of the strides to use on the training set to move the patch on the image. 
        patch_size = None, int.
            Size of the square patches that are used to cut the image and feed the model.
        
        """
        
        if X is None: X = self.imgs_array
        if patch_size is None: patch_size = self.patch_size
        if strides is None: strides = self.strides
            
        self.imgs_patch = imgs_array_to_imgs_patch_array(X, patch_size, strides)

    
    def split_data(self, X = None, test_ratio = None):
        """
        Description:
        --------------
        Split the data into a training set and a test set.
        
        Parameters:
        --------------
        X = None, np.array
            array of images to process
        test_ratio = None, int.
            Ratio of the training set to use for testing, at the end of the training. (test_ratio < 1).
        
        """
        print("spliting dataset.")
        
        if X is None: X = self.imgs_patch
        if test_ratio is None: test_ratio = (self.test_ratio or 0)
        
        assert (len(X.shape) == 3 or X.shape[2] == 1), "Images must be in grey scale before spliting the dataset. Current shape is {0}".format(X.shape)
        assert test_ratio >= 0 and test_ratio <= 1, "\'test_ratio\' must be within the interval [0;1]"
        
        #TODO : random permutation of the array with seed = 1
        
        X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1))
        if test_ratio > 0:
            i = int((1-test_ratio)*X.shape[0])
            j = X.shape[0] - i        
            self.imgs_train = X[:i, :, :, :]
            self.imgs_test = X[-j:, :, :, :]
        else : 
            self.imgs_train = X
            self.imgs_test = np.array([])
        
        print("{1} has size : {0}".format(X.shape, self.datatype))
        print("{1}_train has size : {0}".format(self.imgs_train.shape, self.datatype))
        print("{1}_test has size : {0}".format(self.imgs_test.shape, self.datatype))
        
    
    def display_images(self, obj = None, range_ = None, mod = None):
        
        """
        Description :
        --------------
        Dispaly set of images. By default, displays images from input.
        
        Parameters :
        --------------
        img_obj = None, np.array.
            array of images.
        range_ = None, range
            range in which we select a subset of indices.
        mod = None, int
            period of the selected images to display 
        """
        if obj is None: obj = self.imgs_array
        if range_ is None: range_ = range(obj.shape[0])
        if mod is None: mod = int(obj.shape[0]/10)
            
        mod = max(4, mod)
        for img, i  in enumerate(range_):
            if i%mod == 0:
                plt.figure(figsize = (10, 4))
                plt.subplot(1,4,1)
                plt.imshow(np.squeeze(obj[i]), cmap='Greys_r')
                plt.title("Image {0}".format(i))
                plt.subplot(1,4,2)
                plt.imshow(np.squeeze(obj[i+1]), cmap='Greys_r')
                plt.subplot(1,4,3)
                plt.imshow(np.squeeze(obj[i+2]), cmap='Greys_r')
                plt.subplot(1,4,4)
                plt.imshow(np.squeeze(obj[i+3]), cmap='Greys_r')
                plt.show()

        
        
class LoadTestSet(ProcessDataSet):
    """
    Derived class of ProcessDataSet made for Test Set.
    """

    def load_images(self, imgs_path = None, dataset_length = 'all'):
        """
        Description:
        --------------
        Load images for test set from directories.
        
        Parameters:
        --------------
        imgs_path = None, string.
            relative path to the image folder.
        dataset_length = 'all', int.
            Length of the data set to load. By default, 'all' loads the full data set.
        """
        
        if imgs_path is None: imgs_path = self.imgs_path
        if self.dataset_length is not None: dataset_length = self.dataset_length
            
        assert imgs_path is not None, "Must specify a relative path for dataset"
        
        self.repositories = sorted(os.listdir(imgs_path))
        self.dataset_length = len(self.repositories) if dataset_length == 'all' else min(dataset_length, len(self.repositories))

        file = os.listdir(imgs_path + self.repositories[0])

        image = load_image(imgs_path + self.repositories[0]+'/'+file[0])
        self.imgs_array = np.zeros((self.dataset_length, image.shape[0], image.shape[1], image.shape[2]))

        for i, repo in enumerate(self.repositories[:self.dataset_length]):
            file = os.listdir(imgs_path + repo)
            assert len(file) == 1, "Each directory must contain only one picture."
            image = load_image(imgs_path + repo + '/' + file[0])
            self.imgs_array[i,:,:] = image
    
    def dataset_augmentation(self, X = None):
        
        if X is None: X = self.imgs_array
    
    def process_data(self):
        """
        Description : 
        --------------
        Process the input data to feed the model.
        """
        self.image_preprocessing()
        self.dataset_augmentation()
        self.resize_dataset_to_fit_model()
        self.split_data(test_ratio = 1)
        #self.display_images(obj = self.imgs_patch, range_ = range(1))
        
        
class LoadTrainSet(ProcessDataSet):
    """
    Derived class of ProcessDataSet made for Train Set.
    """

    def load_images(self, imgs_path = None, dataset_length = 'all'):
        """
        Description:
        --------------
        Load images for test set from directories.
        
        Parameters:
        --------------
        imgs_path = None, string.
            relative path to the image folder.
        dataset_length = 'all', int.
            Length of the data set to load. By default, 'all' loads the full data set.
        """
        
        if imgs_path is None: imgs_path = self.imgs_path
        if self.dataset_length is not None: dataset_length = self.dataset_length
            
        assert imgs_path is not None, "Must specify a relative path for dataset"
        
        files = os.listdir(imgs_path)
        self.dataset_length = len(files) if dataset_length == 'all' else min(dataset_length, len(files))
        
        imgs = [load_image(imgs_path + files[i]) for i in range(self.dataset_length)]
        self.imgs_array = np.asarray(imgs)
        
    def process_data(self, test_ratio = None):
        """
        Description : 
        --------------
        Process the input data to feed the model.
        
        Parameters : 
        --------------
        test_ratio = None, int.
            Ratio of the training set to use for testing, at the end of the training. (test_ratio < 1).
        """
        if test_ratio is None: test_ratio = self.test_ratio
            
        self.image_preprocessing()
        self.dataset_augmentation()
        self.resize_dataset_to_fit_model()
        self.split_data(test_ratio = test_ratio)
        #self.display_images(obj = self.imgs_patch, range_ = range(1))
        
    def dataset_augmentation(self, X = None, vertical_flip = None, horizontal_flip = None, random_rotation = None):
        """
        Description:
        -------------
        Performs a data augmentation.
        
        Parameters:
        -------------
        vertical_flip = None, bool
            Apply vertical flip to the training set before training for data augmentation.
        horizontal_flip = None, bool.
            Apply horizontal flip to the training set before training for data augmentation.
        random_rotation = None, int.
            Number of random rotations per image to apply to the training set before training for data augmentation.
        """
        if X is None: X = self.imgs_array
        if vertical_flip is None: vertical_flip = self.vertical_flip
        if horizontal_flip is None: horizontal_flip = self.horizontal_flip
        if random_rotation is None: random_rotation = self.random_rotation
        
        
        assert (self.datatype == 'images' or self.datatype == 'masks'), 'datatype must be one of \'images\', \'masks\''
        
        print("Performing data augmentation...")
        
        np.random.seed(1)
        angle_rot = numpy.random.randint(0, 90, (random_rotation, len(X)))
        self.imgs_array = X
        
        if horizontal_flip:
            imgs_horizontal_flip = np.flip(X, axis =  2)
            self.imgs_array = np.concatenate((self.imgs_array, imgs_horizontal_flip), axis = 0)
        if vertical_flip:
            imgs_vertical_flip = np.flip(X, axis =  1)
            self.imgs_array = np.concatenate((self.imgs_array, imgs_vertical_flip), axis = 0)

        
        
        for rot in angle_rot:
            imgs_rot = np.zeros(X.shape)
            for i, x in enumerate(X):
                imgs_rot[i, :, :] = rotate(x, angle = rot[i], mode='reflect', axes=(0,1), reshape = False, order = 5)
            if self.datatype == 'images':
                imgs_rot = improve_image_contrast(imgs_rot)
            if self.datatype == 'masks' :
                imgs_rot = contineous_to_binary_mask(imgs_rot)

            self.imgs_array = np.concatenate((self.imgs_array, imgs_rot), axis = 0)
            
        print("Dataset augmentation shape : {0} -> {1}".format(X.shape, self.imgs_array.shape))
