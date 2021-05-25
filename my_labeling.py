__authors__ = 'TO_BE_FILLED'
__group__ = 'TO_BE_FILLED'

import numpy as np
import Kmeans
import KNN
from utils_data import read_dataset, visualize_k_means, visualize_retrieval
import matplotlib.pyplot as plt
import cv2
import os


def retrieval_by_shape(images, labels, question):
    images_matching = []

    for i, class_label in enumerate(labels):
        if question == class_label:
            images_matching.append(images[i])

    return np.array(images_matching)


def train_program():
    # Train classes
    train_classes_num = round(0.20 * train_imgs.shape[0])  # Limit to 20% of images
    knn = KNN.KNN(train_imgs[:train_classes_num],
                  train_class_labels[:train_classes_num])

    # Test classes
    test_classes_num = round(0.80 * test_imgs.shape[0])  # Limit to 70% of test classes
    class_labels = knn.predict(test_imgs[:test_classes_num], 5)

    return train_classes_num, test_classes_num, class_labels


if __name__ == '__main__':
    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, \
    test_imgs, test_class_labels, test_color_labels = read_dataset(ROOT_FOLDER='./images/', gt_json='./test/gt.json')

    # List with all the existant classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    train_class_num, test_classes_num, class_labels = train_program()

    results = retrieval_by_shape(test_imgs[:test_classes_num], class_labels, "Skirt")
    visualize_retrieval(results, 10)


