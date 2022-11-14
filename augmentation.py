import numpy as nd
import random
from noise import noise, normalize

# adds N blank spectrogram / blank label tuples to dataset (D) for training silent
# input behavior
def add_silence_to_dataset(D, N):
    c_ids = ''

    for index in range(0, N):
        D.append((nd.zeros((128, 128), dtype = nd.float32), c_ids))

    return D


# add some randomly colored, randomly gained noise to add to a 1D audio file prior
# to running it through analysis methods. especially useful for silence training and
# data augmentation systems
def add_noise(y, passes=1):
    N = len(y)

    types = ['white', 'pink', 'blue', 'brown', 'violet']

    n = nd.zeros(N)

    # recursively generate and add noise to n |passes| times
    for i in range(0, passes):
        # select a random noise color algorithm to use
        selected_type = random.randint(0, 4)

        n += noise(N, color=types[selected_type]) / passes

    n = normalize(n)

    n *= nd.random.uniform(0, 0.20)

    y += n

    return y
