__authors__ = ['1571610', '1562750', '1565479']
__group__ = 'DM.18'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist

class KNN:
    PIXELS_PER_DIMENSION = 4800 * 3

    def __init__(self, train_data, labels):

        self._init_train(train_data)
        self.labels = np.array(labels)

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        arr = np.array(train_data)
        if arr.dtype != "float64":
            arr = np.asarray(arr, dtype=np.float64)

        # 4800 = numero de pixels de les imatges, 3 = espai dimensional de colors
        arr = np.reshape(arr, (arr.shape[0], self.PIXELS_PER_DIMENSION))

        self.train_data = arr


    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data:   array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
        :param k:  the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        labels = []
        test_data = np.asarray(test_data, dtype=np.float64)
        self.test_data = np.reshape(test_data, (test_data.shape[0], self.PIXELS_PER_DIMENSION))
        dist = cdist(self.test_data, self.train_data, 'euclidean')

        for i in dist:
            men = i.argsort()[:k]
            labels.append(self.labels[men])

        self.neighbors = np.array(labels)


    def get_class(self):
        """
        Get the class by maximum voting
        :return: 2 numpy array of Nx1 elements.
                1st array For each of the rows in self.neighbors gets the most voted value
                            (i.e. the class at which that row belongs)
                2nd array For each of the rows in self.neighbors gets the % of votes for the winning class
        """
        most_repeated = []

        for neighbor in self.neighbors:
            aux, i, rep = np.unique(neighbor, return_index=True, return_counts=True)
            aux_sort = np.argsort(i)
            i = np.argmax(rep[aux_sort])
            most_repeated.append(aux[aux_sort][i])

        return most_repeated

    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
        :param k:         :param k:  the number of neighbors to look at
        :return: the output form get_class (2 Nx1 vector, 1st the classm 2nd the  % of votes it got
        """
        self.get_k_neighbours(test_data, k)
        return self.get_class()
