from keras.preprocessing.image import ImageDataGenerator as IDG
from keras.utils import to_categorical


def data_generator(X, y, num_classes, batch_size=32):
    return IDG(rescale=1. / 225, shear_range=0.2, zoom_range=0.2, horizontal_flip=True) \
        .flow(X, to_categorical(y, num_classes), batch_size=batch_size)
