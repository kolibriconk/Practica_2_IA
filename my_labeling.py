__authors__ = ['1571610', '1562750', '1565479']
__group__ = 'DM.18'

import time

import numpy as np
import Kmeans
import KNN
from utils_data import read_dataset, visualize_retrieval
import cv2


def train_program():
    # Train classes
    train_classes_num = round(train_imgs.shape[0])  # Limit to 20% of images
    knn = KNN.KNN(train_imgs[:train_classes_num],
                  train_class_labels[:train_classes_num])

    # Test classes
    test_classes_num = round(test_imgs.shape[0])  # Limit to 70% of test classes
    class_labels = knn.predict(test_imgs[:test_classes_num], 6)

    return train_classes_num, test_classes_num, class_labels


def retrieval_by_shape(images, labels, question):
    images_matching = []

    for i, class_label in enumerate(labels):
        if question == class_label:
            images_matching.append(images[i])

    return np.array(images_matching)


def get_shape_accuracy(labels, gt_labels):
    total = len(labels)
    correct = np.sum(labels == gt_labels)

    return correct/total*100


if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, \
    test_imgs, test_class_labels, test_color_labels = read_dataset(ROOT_FOLDER='./images/', gt_json='./test/gt.json')

    # List with all the existant classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))
    startTime = time.time()
    train_class_num, test_classes_num, class_labels = train_program()
    endTime = time.time()
    print("Training class time : {:.2f}".format(endTime - startTime))

    results = retrieval_by_shape(test_imgs[:test_classes_num], class_labels, "Jeans")

    print("Retrieval by shape completed time was: {}".format(endTime-startTime))

    percent = get_shape_accuracy(class_labels, test_class_labels[:test_classes_num])
    print("The % of the shape accuracy is {}%".format(round(percent, 2)))

    visualize_retrieval(results, 12)


