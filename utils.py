from keras.preprocessing.image import ImageDataGenerator as IDG
from keras.utils import to_categorical

def data_generator(X, y, num_classes, batch_size = 32):
    """
        Args:
            X: labels
            y: target
            num_classes: number of classes to predict
            batch_size: batch size (default = 32)

        Returns a keras data generator

    """
    return IDG(rescale=1. / 225, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)\
        .flow(X, to_categorical(y, num_classes), batch_size = 32)
