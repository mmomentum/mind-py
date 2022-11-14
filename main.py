import pandas as pd
import librosa
import librosa.display
import numpy as np
import math
from keras import models, layers
from tensorflow.keras.utils import to_categorical
import random

import convert_model
import augmentation
import preprocessing as pre
import statistics as stats

# characterization strings for the "pad / atmosphere" model
activations = ['intense', 'soft', 'dark', 'bright', 'dense', 'sparse',
               'synthetic', 'organic', 'smooth', 'dynamic']


# modifies |data| object to fill in the 'c_ids' column. also checks if all of the
# filenames are valid by attempting to import a small chunk of them.
def import_and_preprocess_csv():
    data = pd.read_csv('D:/programming/python/mynd-control/pad_model/dataset.csv')

    data['c_ids'] = str()  # add c_ids column

    for row in range(data.shape[0]):
        # add .wav suffix to filename if it is left out in entry
        data['file_name'][row] = pre.add_suffix(data['file_name'][row], '.wav')

        # determine characterization ID values
        data['c_ids'][row] = pre.activations2int(activations,
                                                 data['characterizations'][row])

        # this won't work if the file_name is incorrect or the file doesnt exist
        librosa.load('D:/programming/python/mynd-control/pad_model/data/' +
                     data['file_name'][row], duration=0.05)

    print("Done. Data is formatted as such:")
    print(data.head())

    return data

# runs through the files specified by the CSV and returns an array of spectrogram
# matrices. this process takes awhile so sit back and relax
def import_and_augment_data(inputted_csv, output, count=-1, augment=True):
    # duration in seconds required for a 128^2 spectrogram output
    duration = 2.97 * 0.75

    rows_to_use = 0

    if count == -1:
        rows_to_use = inputted_csv.shape[0] - 1
    else:
        rows_to_use = count

    # run through each file the normal way
    if not augment:
        for row in range(rows_to_use):
            print('Loading ' + inputted_csv['file_name'][row])
            y, sr = librosa.load('D:/programming/python/mynd-control/pad_model/data/'
                                 + inputted_csv['file_name'][row], duration=duration)
            ps = librosa.feature.melspectrogram(y=y, sr=sr)
            output.append((ps, inputted_csv['c_ids'][row]))

    # run through each file and perform data augmentation methods
    else:
        for row in range(rows_to_use):
            y, sr = librosa.load('D:/programming/python/mynd-control/pad_model/data/'
                                 + inputted_csv['file_name'][row])
            file_length_seconds = len(y) / sr

            # sample duration
            samples = math.ceil(sr * duration)

            # how many slices we've actually done
            actual_slices = 0
            audio_processes = 0

            # run through the y recursively to get more data
            for y_slice in range(0, math.floor(file_length_seconds / duration)):
                if y_slice < 12:  # do it up to 3 times to prevent over-fitting issues
                    actual_slices += 1
                    sample_offset = math.ceil((duration * y_slice) * sr)
                    y2 = y[sample_offset:sample_offset + samples]

                    ps = librosa.feature.melspectrogram(y=y2, sr=sr)
                    output.append((ps, inputted_csv['c_ids'][row]))

                    y2 = augmentation.add_noise(y2, passes=random.randint(1, 3))
                    ps = librosa.feature.melspectrogram(y=y2, sr=sr)
                    output.append((ps, inputted_csv['c_ids'][row]))

                    # now that we have done analysis of the dry slice, we process it
                    # slightly to increase our data size (len(shift_factors) +
                    # len(stretch_factors) more data points)

                    shift_factors = [-2.2, 2.2]

                    # pitch shift
                    for i in range(len(shift_factors)):
                        y_augment = librosa.effects.pitch_shift(y2, sr,
                                                                n_steps=shift_factors[i])

                        ps = librosa.feature.melspectrogram(y=y_augment, sr=sr)
                        output.append((ps, inputted_csv['c_ids'][row]))

                        y_augment = augmentation.add_noise(y2,
                                                           passes=random.randint(1, 3))
                        ps = librosa.feature.melspectrogram(y=y_augment, sr=sr)
                        output.append((ps, inputted_csv['c_ids'][row]))

                    # if we stretch over 1.0 we don't have enough samples to run through
                    stretch_factors = [0.9, 0.85]

                    # time stretch
                    for i in range(len(stretch_factors)):
                        y_augment = librosa.effects.time_stretch(y2,
                                                                 rate=stretch_factors[i])
                        y_augment = y_augment[:samples]  # trim it back

                        ps = librosa.feature.melspectrogram(y=y_augment, sr=sr)
                        output.append((ps, inputted_csv['c_ids'][row]))

                        y_augment = augmentation.add_noise(y2,
                                                           passes=random.randint(1, 3))
                        ps = librosa.feature.melspectrogram(y=y_augment, sr=sr)
                        output.append((ps, inputted_csv['c_ids'][row]))

                    # determine number of audio processes for console output
                    audio_processes = len(shift_factors) + len(stretch_factors) + 1

            print('Generated ' + (str(actual_slices * audio_processes)) +
                  ' spectrogram matrices via data augmentation of ' + inputted_csv
                  ['file_name'][row])

    # add silence audio files equal to 1/10th or less of the dataset for optimal training
    samples = math.ceil(22050 * duration)

    '''
    noise_to_add = math.ceil(len(output) / 12)

    for row in range(0, noise_to_add):
        y = np.zeros(samples)
    
        ps = librosa.feature.melspectrogram(y=y, sr=22050)
        D.append((ps, ''))
    
        y = augmentation.add_noise(y)
        ps = librosa.feature.melspectrogram(y=y, sr=22050)
        D.append((ps, ''))

    print('Added ' + str(noise_to_add * 2) +
          ' blank spectrogram matrices at the end of dataset for silence training')
'''
    # return dataset


# prepare for training
def training_preparation(dataset_in):
    dataset = dataset_in

    random.shuffle(dataset)

    train = dataset[:7000]
    test = dataset[7000:]

    X_train, y_train = zip(*train)
    X_test, y_test = zip(*test)

    # Reshape for input
    X_train = np.array([x.reshape((128, 128, 1)) for x in X_train])
    X_test = np.array([x.reshape((128, 128, 1)) for x in X_test])

    # perform encoding on labels
    y_train = get_labels_from_data(train)
    y_test = get_labels_from_data(test)


# return a keras compatible label set from the data
def get_labels_from_data(data_array):
    # we start by making the array via to_categorical()
    input_array = to_categorical([0] * len(data_array), num_classes=len(activations))

    for data_row in range(len(data_array)):
        input_array[data_row][0] = 0  # cheeky little kludge

        # get and split the c_ids into an integer array
        activation_ints = list(map(int, data_array[data_row][1].split()))

        # set the activation labels accordingly. in this case, we run through 'c_ids'
        # and start decreasing the label value as we have more and more labels
        # (i.e [1, 1, 0.5, 0.3, 0.25] for five c_ids).
        for activation in range(len(activation_ints)):
            if activation == 0:
                input_array[data_row][activation_ints[activation]] = 1
            if activation != 0:
                N = len(activation_ints)
                input_array[data_row][activation_ints[activation]] = \
                    (N / activation) / N

    return input_array


# generate and compile neural net
def get_model():
    model = models.Sequential()
    input_shape = (128, 64, 1)

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

    model.add(layers.Dense(10, activation="sigmoid"))

    model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=['accuracy'])

    return model


# different convolutional model with higher complexity than the first
def get_model_v2():
    num_filters = [24, 32, 64, 128]
    pool_size = (2, 2)
    kernel_size = (3, 3)
    input_shape = (128, 64, 1)
    num_classes = 10

    model = models.Sequential()

    model.add(layers.Conv2D(num_filters[0], kernel_size, padding="same",
                            input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=pool_size))

    model.add(layers.Conv2D(num_filters[1], kernel_size, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=pool_size))

    model.add(layers.Conv2D(num_filters[2], kernel_size, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=pool_size))

    model.add(layers.Conv2D(num_filters[3], kernel_size, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))

    model.add(layers.GlobalMaxPooling2D())
    model.add(layers.Dense(num_filters[3], activation="relu"))
    model.add(layers.Dense(num_classes, activation="sigmoid"))

    model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=['accuracy'])

    return model


def get_model_rnn():
    input_shape = (128, 64)
    num_classes = 10

    model = models.Sequential()
    model.add(layers.GRU(128, input_shape=input_shape))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(num_classes, activation="sigmoid"))

    model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=['accuracy'])

    return model


def get_model_v3():
    nb_filters = 32
    pool_size = (2, 2)
    kernel_size = (3, 3)
    nb_layers = 4
    input_shape = (128, 96, 1)

    model = models.Sequential()

    model.add(layers.Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        padding="same", input_shape=input_shape))
    model.add(layers.BatchNormalization(axis=1))
    model.add(layers.Activation('relu'))

    for layer in range(nb_layers - 1):
        model.add(layers.Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
        model.add(layers.BatchNormalization(axis=1))
        model.add(layers.ELU(alpha=1.0))
        model.add(layers.MaxPooling2D((4, 2), strides=(4, 2)))
        model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(128))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10))
    model.add(layers.Activation("sigmoid"))

    return model


# broken currently?
def export_for_fdeep(model):
    model.save_weights('tmp_exported_model')
    convert_model.convert('tmp_exported_model', '../out/output_temp.json')
