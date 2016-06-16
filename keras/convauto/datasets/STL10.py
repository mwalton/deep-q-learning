from __future__ import absolute_import
import numpy as np
import os, sys, tarfile, urllib

# image shape
HEIGHT = 96
WIDTH = 96
DEPTH = 3

SIZE = HEIGHT * WIDTH * DEPTH
DATA_DIR = './data'
DATA_URL = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'
DL_PATH = './data/stl10_binary.tar.gz'
DATA_PATH = './data/stl10_binary/train_X.bin'
LABEL_PATH = './data/stl10_binary/train_y.bin'
FOLD_PATH = './data/stl10_binary/fold_indices.txt'
TEST_PATH ='./data/stl10_binary/test_X.bin'
TEST_LABEL_PATH = './data/stl10_binary/test_y.bin'
UNLAB_PATH = './data/stl10_binary/unlabeled_X.bin'

def download_and_extract():
    """
    Download and extract the STL-10 dataset
    :return: None
    """
    dest_directory = DATA_DIR
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not (os.path.exists(filepath) or os.path.exists(DATA_PATH)) :
        def _progress(count, block_size, total_size):
            sys.stdout.write('\rDownloading %s %.2f%%' % (filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.urlretrieve(DATA_URL, filepath, reporthook=_progress)
        print('Downloaded', filename)
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def read_labels(path_to_labels):
    """
    :param path_to_labels: path to the binary file containing labels from the STL-10 dataset
    :return: an array containing the labels
    """
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels

def read_data(path_to_data):
    """
    :param path_to_data: path to the binary file containing data from the STL-10 dataset
    :return: an array containing the images in column-major order
    """
    with open(path_to_data, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)
        # We force the data into 3x96x96 chunks, since the
        # images are stored in "column-major order", meaning
        # that "the first 96*96 values are the red channel,
        # the next 96*96 are green, and the last are blue."
        # The -1 is since the size of the pictures depends
        # on the input file, and this way numpy determines
        # the size on its own.
        images = np.reshape(everything, (-1, 3, 96, 96))
        return images

def read_fold_index(path_to_folds):
    """
    :param path_to_folds: path to the text file containing fold indices.
    Each line corresponds to a fold.
    :return: a list containing the index sets for each fold.
    """
    fold_set = []
    with open(path_to_folds, "r") as f:
        fold_lists = f.readlines()
        for index, item in enumerate(fold_lists):
            raw_list = item.split(" ")
            del raw_list[-1]
            int_list = [ int(x) for x in raw_list ]
            fold_set.append(int_list)
        return fold_set

def load_fold_indices(fnum):
    """
    :param fnum: which fold to load.
    :return: a list containing the index sets for fold fnum.
    """
    fold_set = read_fold_index(FOLD_PATH)
    fold_indices = fold_set[fnum]
    return fold_indices

def load_fold(fnum):
    """
    :param fnum: which fold to load.
    :return: a train / test dataset based on a specific STL10 fold.
    """
    download_and_extract()
    fold_set = read_fold_index(FOLD_PATH)
    fold_indices = load_fold_indices(fnum)

    nb_fold_samples = len(fold_indices)

    X_train = np.zeros((nb_fold_samples, DEPTH, HEIGHT, WIDTH), dtype="uint8")
    y_train = np.zeros((nb_fold_samples,), dtype="uint8")

    images = read_data(DATA_PATH)
    labels = read_labels(LABEL_PATH)

    X_train = images[fold_indices, :, :, :]
    y_train = labels[fold_indices] -1

    test_images = read_data(TEST_PATH)
    test_labels = read_labels(TEST_LABEL_PATH)

    nb_test_samples = len(test_labels)

    X_test = np.zeros((nb_test_samples, DEPTH, HEIGHT, WIDTH), dtype="uint8")
    y_test = np.zeros((nb_test_samples,), dtype="uint8")

    X_test = test_images
    y_test = test_labels-1

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    return (X_train, y_train), (X_test, y_test)

def load_unlabeled():
    """
    :return: a train / test dataset based on a the unlabeled STL10 set.
    Suitable for use in autoencoders and other feature extraction methods.
    """
    download_and_extract()
    images = read_data(UNLAB_PATH)
    nb_unlab_samples = len(images)

    X_train = np.zeros((nb_unlab_samples, DEPTH, HEIGHT, WIDTH), dtype="uint8")
    y_train = np.zeros((nb_unlab_samples,), dtype="uint8")

    X_train = images
    y_train = images

    return (X_train, y_train)


def load_labeled():
    """
    :return: a train / test dataset based on all available labels from STL10
    Suitable for use in more traditional machine learning experiments.
    """
    download_and_extract()

    images = read_data(DATA_PATH)
    labels = read_labels(LABEL_PATH)

    nb_samples = len(labels)

    X_train = np.zeros((nb_samples, DEPTH, HEIGHT, WIDTH), dtype="uint8")
    y_train = np.zeros((nb_samples,), dtype="uint8")

    X_train = images
    y_train = labels -1

    test_images = read_data(TEST_PATH)
    test_labels = read_labels(TEST_LABEL_PATH)

    nb_test_samples = len(test_labels)

    X_test = np.zeros((nb_test_samples, DEPTH, HEIGHT, WIDTH), dtype="uint8")
    y_test = np.zeros((nb_test_samples,), dtype="uint8")

    X_test = test_images
    y_test = test_labels -1

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    return (X_train, y_train), (X_test, y_test)


def plot_image(encoded_image):
    import matplotlib.pyplot as plt
    image = np.transpose(encoded_image, (2, 1, 0))
    plt.imshow(image)
    plt.show()