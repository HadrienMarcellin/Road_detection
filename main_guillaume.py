#!/usr/bin/env python3

from BBQPL_pipeline import *

if __name__ == '__main__':
    
    pipe = Pipeline(#u_net_file='model_16Filters_TrueHf_TrueVf_3RandRot_128PatchSize5Epochs_ana', 
                u_net_suffix = 'guillaume',
                training_images_dir="training/images/", 
                training_masks_dir="training/groundtruth/", 
                testing_images_dir="test_set_images/",
                predicted_dir = 'results/',
                submission_filename = 'submission',
                training_dataset_length = 'all', 
                testing_dataset_length = 'all',
                training_strides = 34,
                testing_strides = 16,
                patch_size = 128,
                batch_size = 16,
                epochs = 30,
                nb_filters = 64,
                test_ratio = 0, 
                validation_ratio = 0.2,
                horizontal_flip = True,
                vertical_flip = True,
                preprocess = True,
                random_rotation = 2,
                make_prediction = True, 
                make_training = True)
    

