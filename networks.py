from keras.applications import VGG16
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.datasets import cifar10
from keras.engine.topology import get_source_inputs
from keras import Input
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator as IDG
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from kerassurgeon import Surgeon

import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Cifar10VGG16:

    def __init__(self):

        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
        self.model = self.__build_model()
        self.num_classes = 10
        self.obversation_space = None
        self.action_space = None


    def __build_model(self):
        input_shape = self.x_train.shape[1:]
        input_tensor = Input(shape=(input_shape))
        vgg = VGG16(include_top = False, input_tensor = input_tensor, weights='imagenet')
        flatten = Flatten(name='Flatten')(vgg.output)
        prediction = Dense(10, activation='softmax')(flatten)
        model = Model(input_tensor, prediction)
        model.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.01), metrics=['accuracy'])
        return model

    def get_feature_map(self, name):
        model = Model(inputs=self.model.input, outputs=self.model.get_layer(name).output)
        img = image.load_img('nn.png', target_size=(32, 32))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        x = model.predict(x)
        x = x.transpose(3, 0, 1, 2).reshape(x.shape[-1], -1) # reshaping the feature the feature map
        self.obversation_space = np.prod(x)
        self.action_shape = x[0]
        return x


    def evaluate(self):
        eval_data_generator = IDG(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, \
            horizontal_flip=True).flow(self.x_test, to_categorical(self.y_test, self.num_classes))
        results = self.model.evaluate_generator(eval_data_generator, eval_data_generator.n, verbose = 1)
        return results[0] #return the loss

    def train(self):
        train_data_generator = IDG(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, \
            horizontal_flip=True).flow(self.x_train, to_categorical(self.y_train, self.num_classes), \
            batch_size=32, shuffle=True)

        eval_data_generator = IDG(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, \
            horizontal_flip=True).flow(self.x_test, to_categorical(self.y_test, self.num_classes), \
            batch_size=32)

        results = self.model.fit_generator(train_data_generator, \
            train_data_generator.n // train_data_generator.batch_size, \
            1, validation_data = eval_data_generator, \
            validation_steps = eval_data_generator.n // eval_data_generator.batch_size)

        return results



    def step(self, action):
        pass
        # surgeon = Surgeon(self.model)
        # surgeon.add_job('delete_channels', self.model, layer, channels=action)
        # return surgeon.operate()



if __name__ == '__main__':
    model = Cifar10VGG16()
    print(model.get_feature_from_layer('block4_pool'))
