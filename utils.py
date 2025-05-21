import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

def data_generator(X, y, num_classes, batch_size=32):
    """Create a data generator with data augmentation"""
    gen = ImageDataGenerator(
        rescale=1./255 if X.dtype != 'float32' else None,  # Skip rescaling if already normalized
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    return gen.flow(X, to_categorical(y, num_classes), batch_size=batch_size)