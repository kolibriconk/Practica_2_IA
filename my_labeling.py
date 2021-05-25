__authors__ = ['1571610', '1562750', '1565479']
__group__ = 'DM.18'

import numpy as np
import Kmeans
import KNN
import time
import datetime as dt
from utils_data import read_dataset, visualize_k_means, visualize_retrieval
import matplotlib.pyplot as plt
import cv2


def Kmeans_stadistics(Kmeans, Kmax):
    Kmeans = Kmeans()
    for k in Kmax:
        if k > 2:
            Kmeans.fit()
            Kmeans.whitinClassDistance()
            Kmeans.converges()
            start_time = dt.datetime.today().timestamp()

    time_diff = dt.datetime.today().timestamp() - start_time
    iter_sec = Kmax / time_diff

def Get_shape_accuracy():
    


if __name__ == '__main__':

    #Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, \
    test_imgs, test_class_labels, test_color_labels = read_dataset(ROOT_FOLDER='./images/', gt_json='./test/gt.json')

    #List with all the existant classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))
    # Train classes
    n_train_classes = round(0.20 * train_imgs.shape[0])
    knn = KNN.KNN(train_imgs[:n_train_classes],
                  train_class_labels[:n_train_classes])

    # Test classes
    n_test_classes = round(0.80 * test_imgs.shape[0])
    class_labels = knn.predict(test_imgs[:n_test_classes], 5)

    results = retrieval_by_shape(test_imgs[:n_test_classes], class_labels, "")
    visualize_retrieval(results, 10)
    #Kmeans.__init__(test_imgs)
    stadistics = Kmeans_stadistics(KNN, 20)
    visualize_k_means(Kmeans, test_imgs.shape())
    #for ix, input in enumerate(self.test_cases['input']):
      #  km = Kmeans(input, self.test_cases['K'][ix])
      #  e = Kmeans_stadistics(km, 20)
      #  np.testing.assert_array_equal(e, self.test_cases['stadistics'][ix])














