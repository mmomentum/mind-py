from keras import models, layers
from tensorflow.keras.utils import to_categorical


# convolutional neural net with the number of internal layers outside of input/output
# definable via nb_layers (default is 4 but for realtime settings it should be lower)
def multilayer_cnn(X, Y, nb_classes, nb_layers=4):
    nb_filters = 32  # number of convolutional filters = "feature maps"
    kernel_size = (2, 3)  # convolution kernel size
    pool_size = (2, 2)  # size of pooling area for max pooling
    cl_dropout = 0.5  # conv. layer dropout
    dl_dropout = 0.6  # dense layer dropout

    input_shape = (X, Y, 1)
    model = models.Sequential()
    model.add(layers.Conv2D(nb_filters, kernel_size, padding='same',
                            input_shape=input_shape, name="Input"))
    model.add(layers.MaxPooling2D(pool_size=pool_size))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization(axis=-1))

    for layer in range(nb_layers - 1):  # add more layers than just the first
        model.add(layers.Conv2D(nb_filters, kernel_size, padding='same'))
        model.add(layers.MaxPooling2D(pool_size=pool_size))
        model.add(layers.Activation('elu'))
        model.add(layers.Dropout(cl_dropout))

    model.add(layers.Flatten())
    model.add(layers.Dense(128))  # 128 is 'arbitrary' for now
    model.add(layers.Activation('elu'))
    model.add(layers.Dropout(dl_dropout))
    model.add(layers.Dense(nb_classes))
    model.add(layers.Activation("sigmoid"))

    model.compile(optimizer="Adam", loss=["binary_crossentropy",
                                          "sparse_categorical_crossentropy"], metrics=[
        'accuracy'])

    return model


# original model without multiple layers. while it's simple it might be the only neural
# net that i've designed that can work in a realtime setting (only about 1MB worth of
# weights and biases as opposed to 16MB for a 2 layer |multilayer_cnn|).
def simple_cnn(X, Y, nb_classes, ):
    model = models.Sequential()
    input_shape = (X, Y, 1)

    # i think that irregularly shaped convolution kernels would be good for spectrogram
    # data but i don't know how easily they can be implemented
    model.add(layers.Conv2D(24, (5, 5), strides=(1, 1), input_shape=input_shape))
    model.add(layers.MaxPooling2D((4, 2), strides=(4, 2)))
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(48, (5, 5), padding="valid"))
    model.add(layers.MaxPooling2D((4, 2), strides=(4, 2)))
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(48, (5, 5), padding="valid"))
    model.add(layers.Activation('relu'))

    model.add(layers.Flatten())
    model.add(layers.Dropout(rate=0.5))

    model.add(layers.Dense(64))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(rate=0.5))

    model.add(layers.Dense(nb_classes, activation="sigmoid"))

    # dual loss compiled models seem to train very nicely
    model.compile(optimizer="Adam", loss=["binary_crossentropy",
                                          "sparse_categorical_crossentropy"], metrics=[
        'accuracy'])

    return model
