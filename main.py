#!/usr/bin/env python3

from BBQPL_pipeline import *

if __name__ == '__main__':
    
    pipe = Pipeline(#u_net_file='model_16Filters_FalseHf_FalseVf_1RandRot_128PatchSize', 
                training_images_dir="training/images/", 
                training_masks_dir="training/groundtruth/", 
                testing_images_dir="test_set_images/",
                predicted_dir = 'results/',
                submission_filename = 'submission',
                training_dataset_length = 1, 
                testing_dataset_length = 2,
                training_strides = 16,
                testing_strides = 16,
                patch_size = 128,
                batch_size = 16,
                epochs = 5,
                nb_filters = 16,
                test_ratio = 0, 
                validation_ratio = 0.2,
                horizontal_flip=True,
                vertical_flip=True,
                random_rotation=1,
                make_prediction = True, 
                make_training = True)
    

