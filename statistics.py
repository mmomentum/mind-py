import matplotlib.pyplot as plt
import numpy as np

def plot_characterization_frequency(activations, data):
    # first we make a big string of all of the 'c_id' cells
    all_cids = ''
    for row in range(data.shape[0]):
        all_cids += data['c_ids'][row]

    # and then we determine how many we have of each c_id
    cid_counts = [0] * 10

    for i in range(0, 10):
        cid_counts[i] = all_cids.count(str(i))

    # and then we plot it
    plt.xlabel('Characterizations')
    plt.ylabel('Count')
    plt.bar(activations, cid_counts)
    plt.show()


def plot(index):
    plt.clf()
    plt.ylim(0, 1)
    plt.bar(np.arange(10), y_test[index], color=(0.2, 0.4, 0.6, 0.4))
    plt.bar(np.arange(10), preds[index], color=(0.8, 0.6, 0.4, 0.4))


