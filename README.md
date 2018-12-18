# Road detection from satellite images

In the scope of this project, we aim to analyse satellite images to detect with precision the location of the roads. To this end, we use well known Machine Learning and Image Processing algorithms.

This project is fully implemented in python, we use many external libraries to compute heavy calculations. In order to run the code and reproduce the results, you should setup your environment to meet the numerous dependencies listed in the next section. 
The provided code allows you to create and train a new model given a bench of input parameters and input training set. You can also use an existing model and test its prediction on any other set of images.

## Dependencies

The project depends on many external libraries that help us compute tough tasks. Here is the list of which your environment must satisfy before you run the code.

- `numpy` : Allows different kind of operations on multi-dimensional arrays. 
- `imageio`: Allows to load and save images.
- `Keras` : Allows a simplified use of 'Tensorflow' library.   
- `Tensorflow` : Allows to model and train a neural network
- `matplotlib` : Allows to display results under graphs or images. It ùalso provide tools to load images.
- `tqdm`: Allows to display loading bars in 'for' loop.
- `scikit-image` : Provides tools to process rgb and greyscale images.  
- `scipy`: Provides tools to save images.
- `sklearn` : Provides computations for f1 score and other metrics.
- `pickle` : Provides tools to save the history of the trained model.


## Run

You can now clone and run the project. To that end, you may download a set of images for the training and a set of images for the testing on the crowdAI plateform. Please note the structure of the training set and the testing set. They are respectively stored in two different patterns. In the training folder you have two folders 'groundtruth/' and 'images/' which contains all the masks and satellite images. In the testing folder, you have one image per folder, hence as much folder as images. The structure of your directory must be the same to run the project.  
  
You will find in the run.py a call function for the initialisation of an object of the class Pipeline with the following parameters :

- `u_net_file` : (default None)
- `u_net_suffix` : (default = '')
- `training_images_dir`  :(default = None) 
- `training_masks_dir` : (default = None) 
- `testing_images_dir` : (default = None)
- `predicted_dir` : (default = None)
- `submission_filename` : (default = None)
- `training_dataset_length` : (default = 'all'), 
- `testing_dataset_length` : (default = 'all'),
- `training_strides` = (default = '16')
- `testing_strides` = (default = 16)
- `patch_size` : (default = 128)
- `batch_size` : (default = 16)
- `epochs` : (default = 5)
- `nb_filters` : (default = 16)
- `test_ratio` : (default =  0) 
- `validation_ratio` : (default =  0.2)
- `horizontal_flip` : (default =  False)
- `vertical_flip` : (default =  False)
- `preprocess` : (default =  False)
- `random_rotation` : (default =  0)
- `make_prediction` : (default =  True)
- `make_training` : (default =  True) 


You may edit this file and tune the parameters of interest to change the training of the model.

