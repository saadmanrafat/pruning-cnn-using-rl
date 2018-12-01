from keras.applications import VGG16
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.datasets import cifar10

from keras import Input
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator as IDG
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from kerassurgeon.operations import delete_channels

import os
import math
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Cifar10VGG16:

    def __init__(self, layer_name = None, b = 0.5):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
        self.model = self.__build_model()
        self.num_classes = 10
        self.layer_name = layer_name or 'block5_conv1'
        self.b = b
        self.action_size = None
        self.state_size = None
        self.epochs = 2

    def __build_model(self):
        """Builds the VGG16 Model
        """
        input_shape = self.x_train.shape[1:]
        input_tensor = Input(shape=(input_shape))
        vgg = VGG16(include_top = False, input_tensor = input_tensor, weights='imagenet')
        flatten = Flatten(name='Flatten')(vgg.output)
        prediction = Dense(10, activation='softmax')(flatten)
        model = Model(input_tensor, prediction)
        model.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.01), metrics=['accuracy'])
        return model

    def get_feature_map(self):
        """Returns a feature maps of a specified layers
        """
        model = Model(inputs=model.input, outputs=model.get_layer(self.layer_name).output)
        img = image.load_img('nn.png', target_size=(32, 32))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        x = model.predict(x)
        x = x.transpose(3, 1, 2, 0).reshape(x.shape[3], -1)
        self.action_size, self.state_size = x.shape
        return x


    def _accuracy_term(self, other):
        """Compares the baseline model with the newly optimized model

        Returns: The accuracy term.
        """
        eval_data_generator = IDG(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, \
            horizontal_flip=True).flow(self.x_test, to_categorical(self.y_test, self.num_classes), \
            batch_size=32)
        train_data_generator = IDG(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, \
            horizontal_flip=True).flow(self.x_train, to_categorical(self.y_train, self.num_classes), \
            batch_size=32, shuffle=True)
        print('Training the optimized model.')
        other.fit_generator(train_data_generator, train_data_generator.n // train_data_generator.batch_size, \
            epochs=self.epochs, validation_data = eval_data_generator, validation_steps = eval_data_generator.n // eval_data_generator.batch_size)
        print('Evaluating the optimized model')
        p_hat = other.evaluate_generator(eval_data_generator, eval_data_generator.n, verbose = 1)[0]
        print('Calculating the accuracy of the base line model')
        p_star = self.model.evaluate_generator(eval_data_generator, eval_data_generator.n, verbose = 1)[0]
        print('Calculating the accuracy term')
        accuracy_term = (self.b - (p_star - p_hat)) / self.b
        return accuracy_term


    def step(self, action):
        """Inputs: the values returned from the neural network an array of {1, 0} values
            signifying the importance of each feature map

            Returns: Action, Reward, Current State, New State
        """
        # finding indexes of the all the unimportant feature maps
        action = np.where(action == 0)[0]
        print('Deleting Channels')
        new_model = delete_channels(self.model, layer = self.model.get_layer(self.layer_name), channels = action)
        print('Calulating Rewards')
        reward = self._accuracy_term(new_model) - math.log10(self.action_size/len(action))
        current_state = self.get_feature_map()
        return action, reward, current_state, True
