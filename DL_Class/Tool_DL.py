import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os

import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords


def normalization(a1):
    return (a1 - a1.min()) / (a1.max() - a1.min())


def show_training(training, var, model_name=None):
    if model_name is None:
        file_name = [f'Untitled_{i}' for i in var]
    else:
        file_name = [f'{model_name}_{i}' for i in var]

    for i, v in enumerate(var):
        plt.plot(training.history[v])
        plt.title(model_name)
        plt.ylabel(i)
        plt.xlabel('Epoch')
        plt.savefig(f'Result/{file_name[i]}')
        plt.show()


def testing(model, x, y):
    # return loss, acc
    return model.evaluate(x, y, verbose=2)


def get_mnist(train_size=None, seed=77):
    (train_x, train_y), (test_x, test_y) = mnist.load_data()

    if train_size:
        x = np.concatenate((train_x, test_x))
        y = np.concatenate((train_y, test_y))
        train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=train_size, random_state=seed)

    train_x = np.expand_dims(train_x, 3) / 255
    test_x = np.expand_dims(test_x, 3) / 255
    train_y = tf.one_hot(train_y, 10)
    test_y = tf.one_hot(test_y, 10)

    return (train_x, train_y), (test_x, test_y)


def get_stopwords(language='english', del_words=None):
    if del_words is None:
        del_words = ['no', 'nor', 'not', 'don', "don't"]

    stop_w = stopwords.words(language)
    stop_w = stop_w[:stop_w.index('ain')]

    for i in del_words:
        stop_w.remove(i)
    return stop_w


def get_glove6B(dimension=100):
    embeddings_index = {}
    f = open(os.path.join(f'Work2Vec_Transform/glove.6B.{dimension}d.txt'), encoding='utf8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index

