from keras.applications import VGG16
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.datasets import cifar10
from keras.engine.topology import get_source_inputs
from keras import Input

from kerassurgeon import Surgeon

class Cifar10VGG16:

    def __init__(self):

        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
        self.model = self.__build_model()


    def __build_model(self):
        input_shape = self.x_train.shape[1:]
        input_tensor = Input(shape=(input_shape))
        vgg = VGG16(include_top = False, input_tensor = input_tensor, input_shape = input_shape)
        prediction = Dense(10, activation='softmax', name='output_layer')(vgg.output)
        return Model(input_tensor, prediction)


    def get_feature_from_layer(name):
        pass

    def observation_space():
        pass

    def action_space():
        pass


    def step(action):
        surgeon = Surgeon(self.model)
        surgeon.add_job('delete_channels', self.model, layer, channels=action)
        return surgeon.operate()


    def __str__():
        return self.model.summary()


if __name__ == '__main__':
    model = Cifar10VGG16()
    print(model.model.summary())
