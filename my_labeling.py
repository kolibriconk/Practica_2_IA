__authors__ = ['1571610', '1562750', '1565479']
__group__ = 'DM.18'

import time

import numpy as np
import Kmeans as km
import pickle
from Kmeans import *
import KNN
from utils_data import read_dataset, visualize_retrieval
import cv2

def resizeImages(train_imgs, test_imgs):
    #Redimension train images

    train_img_new = []
    for x in train_imgs:
        train_img_new.append(np.array(cv2.resize(x, (0, 0), fx=0.75, fy=0.75)))

    #Redimension test images
    test_img_new = []
    for y in test_imgs:
        test_img_new.append(np.array(cv2.resize(y, (0, 0), fx=0.75, fy=0.75)))

    return np.asarray(train_img_new), np.asarray(test_img_new)

def train_program(train_imgs, test_imgs, pixels_per_dimension):
    # Train classes
    train_classes_num = round(train_imgs.shape[0])  # Limit to 20% of images
    knn = KNN.KNN(train_imgs[:train_classes_num],
                  train_class_labels[:train_classes_num], pixels_per_dimension)

    # Test classes
    test_classes_num = round(test_imgs.shape[0])  # Limit to 70% of test classes
    class_labels = knn.predict(test_imgs[:test_classes_num], 5)
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

def test_find_bestK():
    with open('./test/test_cases_kmeans.pkl', 'rb') as f:
        test_bestK = pickle.load(f)

    tolerance = [0.2, 0.25, 0.17, 0.5, 0.1]
    for ix, input in enumerate(test_bestK['input']):
        km = KMeans(input, test_bestK['K'][ix])
        print('Test ' + str(ix + 1))
        for i in tolerance:
            km.options['tolerance'] = i
            aux = km.find_bestK(10)
            print('Tolerance: ' + str(int(i*100)) + '%')
            print('K = ' + str(km.K))
            print(aux)





if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, \
    test_imgs, test_class_labels, test_color_labels = read_dataset(ROOT_FOLDER='./images/', gt_json='./test/gt.json')

    # List with all the existant classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    startTime = time.time()
    train_class_num, test_classes_num, class_labels = train_program(train_imgs, test_imgs, 4800*3)
    endTime = time.time()
    print("Time with images not resized {:.2f}".format(endTime - startTime))

    train_imgs_new, test_imgs_new = resizeImages(train_imgs, test_imgs)
    startTime = time.time()
    train_class_num, test_classes_num, class_labels = train_program(train_imgs_new, test_imgs_new, 2700*3)
    endTime = time.time()
    print("Time with images resized {:.2f}".format(endTime - startTime))

    results = retrieval_by_shape(test_imgs[:test_classes_num], class_labels, "Jeans")

    print("Retrieval by shape completed")

    percent = get_shape_accuracy(class_labels, test_class_labels[:test_classes_num])
    print("The % of the shape accuracy is {}%".format(round(percent, 2)))

    visualize_retrieval(results, 12)

    test_find_bestK()

    options = {}
    options['km_init'] = 'custom'
    km = KMeans(test_imgs[:test_classes_num], 4, options)
    km._init_centroids()
    #visualize_k_means(km, 4800)


