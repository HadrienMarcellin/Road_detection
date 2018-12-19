# Road detection from satellite images

In the scope of this project, we aim to analyse satellite images to detect with precision the location of the roads. To this end, we use well known Machine Learning and Image Processing algorithms.

This project is fully implemented in python, we use many external libraries to compute heavy calculations. In order to run the code and reproduce the results, you should setup your environment to meet the numerous dependencies listed in the next section. 
The provided code allows you to create and train a new model given a bench of input parameters and input training set. You can also use an existing model and test its prediction on any other set of images.

## Dependencies

The project depends on many external libraries that help us compute tough tasks. Here is the list of which your environment must satisfy before you run the code.

- `python3` : Python is an easy to learn, powerful programming language.
- `numpy` : Allows different kind of operations on multi-dimensional arrays. 
- `Keras` : Allows a simplified use of 'Tensorflow' library.   
- `Tensorflow` : Allows to model and train a neural network
- `matplotlib` : Allows to display results under graphs or images. It ùalso provide tools to load images.
- `scikit-image` : Provides tools to process rgb and greyscale images.  
- `scipy`: Provides tools to save images.
- `sklearn` : Provides computations for f1 score and other metrics.
- `pickle` : Provides tools to save the history of the trained model.


| Package       | Version           | Description  |
| ------------- |:-------------:| -----:|
| python3      | 3.6.6 | Python is an easy to learn, powerful programming language.|
| numpy     | 1.15.4 | Allows different kind of operations on multi-dimensional arrays. |

## Run

You can now clone and run the project. To that end, you may download a set of images for the training and a set of images for the testing on the crowdAI plateform. Please note the structure of the training set and the testing set. They are respectively stored in two different patterns. In the training folder you have two folders 'groundtruth/' and 'images/' which contains all the masks and satellite images. In the testing folder, you have one image per folder, hence as much folder as images. The structure of your directory must be the same to run the project.  
  
You will find in the run.py a call function for the initialisation of an object of the class Pipeline with the following parameters :

- `u_net_file` : (*default* = None). Name of the existing Unet model to load.
- `u_net_suffix` : (*default* = ''). Suffix to add at the end of the model file when saving it.
- `training_images_dir`  :(*default* = None). Relative path to the directory that contains the training data set images. 
- `training_masks_dir` : (*default* = None). Relative path to the directory that contains the training data set masks.
- `testing_images_dir` : (*default* = None). Relative path to the directory that contains the testing data set images.
- `predicted_dir` : (*default* = None). Relative path to the directory that will contain the predicted masks.
- `submission_filename` : (*default* = None). Name of the file to submit to the CrowdAI plateform. The file will be saved as a '.csv'.
- `training_dataset_length` : (*default* = 'all'). Length of the trainig dataset to use. if 'all', take the full dataset.
- `testing_dataset_length` : (*default* = 'all'). Length of the testing dataset to use. if 'all', take the full dataset.
- `training_strides` = (*default* = '16'). Length of the strides to use on the training set to move the patch on the image. 
- `testing_strides` = (*default* = 16). Length of the strides to use on the testing set to move the patch on the image. 
- `patch_size` : (*default* = 128). Size of the square patches that are used to cut the image and feed the model. 
- `batch_size` : (*default* = 16). Batch size for the SGD training alogithm.
- `epochs` : (*default* = 5). Number of epochs for the training.
- `nb_filters` : (*default* = 16). Number of filters the unet model should start with. 
- `test_ratio` : (*default* =  0.2). Ratio of the training set to use for testing, at the end of the training. (test_ratio < 1).
- `validation_ratio` : (*default* =  0.2). Ratio of the training set to use for validation, at the end of each epoch. (validation_ratio < 1).
- `horizontal_flip` : (*default* =  False). Apply horizontal flip to the training set before training for data augmentation.
- `vertical_flip` : (*default* =  False). Apply vertical flip to the training set before training for data augmentation.
- `preprocess` : (*default* =  False). Apply image preprocessing before training.
- `seed` : (*default* = 1). Defines random constant.
- `random_rotation` : (*default* =  0). Number of random rotations per image to apply to the training set before training for data augmentation.
- `make_prediction` : (*default* =  True). Perform prediction if True.
- `make_training` : (*default* =  True). Perform training if True.

You may edit this file and tune the parameters of interest to change the training of the model.

