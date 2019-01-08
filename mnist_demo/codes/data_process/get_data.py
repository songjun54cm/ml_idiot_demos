__author__ = 'JunSong<songjun54cm@gmail.com>'
from tensorflow.examples.tutorials.mnist import input_data
# pip install python-mnist
from mnist import MNIST
from ml_idiot.utils.save_load import data_dump
import os
DATA_HOME = "../../data/"


data_folder = os.path.join(DATA_HOME, 'mnist')
save_data_dir = os.path.join(data_folder, "mnist_data")
if not os.path.exists(save_data_dir):
    os.makedirs(save_data_dir)

# Get the sets of images and labels for training, validation, and
# test on MNIST.
data_sets = input_data.read_data_sets(data_folder)

import numpy as np

mndata = MNIST(data_folder)
train_images, train_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()

train_images = np.reshape(np.array(train_images), (len(train_images), 28, 28))
train_labels = np.array(train_labels)
test_images = np.reshape(np.array(test_images), (len(test_images), 28, 28))
test_labels = np.array(test_labels)
print("train image size: %s" % str(train_images.shape))
print("train label size: %s" % str(train_labels.shape))
print("test image size: %s" % str(test_images.shape))
print("test label size: %s" % str(test_labels.shape))

datas = {
    'train_images': train_images,
    'train_labels': train_labels,
    'test_images': test_images,
    'test_labels': test_labels
}
data_dump(datas, save_data_dir)
import _pickle as pkl
with open(os.path.join(save_data_dir, "mnist.pkl"), "wb") as f:
    pkl.dump(datas, f)

