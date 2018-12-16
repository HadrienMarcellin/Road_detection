#!/usr/bin/env python3

from keras.models import *
from keras.layers import *
from keras.optimizers import *
import pickle

class UnetModel:
    
    def __init__(self, nb_filters = 16, patch_size = 128, horizontal_flip = False, vertical_flip = False, random_rotation = False):
        self.nb_filters = nb_filters
        self.patch_size = patch_size
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.random_rotation = random_rotation
        
    def build_model(self, nb_filters = None, patch_size = None):
        if nb_filters is None: nb_filters = self.nb_filters
        if patch_size is None: patch_size = self.patch_size
            
        input_size = (patch_size, patch_size, 1)
        inputs = Input(input_size)

        conv1 = Conv2D(nb_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = Conv2D(nb_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(nb_filters * 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(nb_filters * 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(nb_filters * 2**2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(nb_filters * 2**2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(nb_filters * 2**3, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(nb_filters * 2**3, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(nb_filters * 2**4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(nb_filters * 2**4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(nb_filters * 2**3, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = concatenate([drop4,up6], axis = 3)
        conv6 = Conv2D(nb_filters * 2**3, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(nb_filters * 2**3, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

        up7 = Conv2D(nb_filters * 2**2, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([conv3,up7], axis = 3)
        conv7 = Conv2D(nb_filters * 2**2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(nb_filters * 2**2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

        up8 = Conv2D(nb_filters * 2, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([conv2,up8], axis = 3)
        conv8 = Conv2D(nb_filters * 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(nb_filters * 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

        up9 = Conv2D(nb_filters, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([conv1,up9], axis = 3)
        conv9 = Conv2D(nb_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(nb_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

        self.model = Model(input = inputs, output = conv10)


    def compile_model(self):
        self.model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
        
    def predict(self, input_data):
        assert input_data is not None, "Cannot predict \'None\' type data. Make sure \'training_dataset_length\' > 0"
        return self.model.predict(input_data)
        
    def train(self, X_input, Y_input, validation_ratio, epochs, batch_size, callbacks):
        
        assert X_input.shape[2] is self.patch_size, " Input images must be of the size of the model."
        
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.history = self.model.fit(X_input, Y_input, validation_split=validation_ratio, epochs=epochs,
                                      batch_size=batch_size, shuffle=True, callbacks=callbacks)
        
        self.save_model()
        
    def save_model(self, suffix = None):
        if suffix is None:
            self.suffix = str(self.nb_filters) + "Filters_" + str(self.horizontal_flip)+ "Hf_" + str(self.vertical_flip) + "Vf_" + str(self.random_rotation) + "RandRot_"  + str(self.patch_size) + "PatchSize" + str(self.epochs) + "Epochs"
            suffix = self.suffix
        
        print("Saving model into file : {0}".format("model_" + suffix))
        self.model.save("model_" + suffix)
        
        print("Saving model's history into file : {0}".format("history_" + suffix))
        with open("history_" + suffix + ".pickle", 'wb') as handle:
            pickle.dump(self.history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def load_model(self, title):
        print("Loading model from file : {0}".format(title))
        self.model = load_model(title)
        
    

