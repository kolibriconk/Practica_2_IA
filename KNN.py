__authors__ = ['1571610', '1562750', '1565479']
__group__ = 'DM.18'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist


class KNN:                             #60*60=3600
    PIXELS_PER_DIMENSION = 4800 * 3  # 4800 = numero de pixels de les imatges, 3 = espai dimensional de colors
    #PIXELS_PER_DIMENSION = 2700 * 3   #60*45=2700
    DATA_TYPE_FLOAT = "float64"
    DISTANCE_TYPE = "euclidean"

    def __init__(self, train_data, labels):

        self._init_train(train_data)
        self.labels = np.array(labels)
        self.neighbors = []

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        arr = np.asarray(train_data)
        if arr.dtype != self.DATA_TYPE_FLOAT:
            arr = np.asarray(arr, dtype=np.float64)

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
        distances = cdist(self.test_data, self.train_data, self.DISTANCE_TYPE)

        for dist in distances:
            men = dist.argsort()[:k]
            labels.append(self.labels[men])


        self.neighbors = np.asarray(labels)

        return self.neighbors

    def get_class(self):
        """
        Get the class by maximum voting
        :return: 2 numpy array of Nx1 elements.
                1st array For each of the rows in self.neighbors gets the most voted value
                            (i.e. the class at which that row belongs)
                2nd array For each of the rows in self.neighbors gets the % of votes for the winning class
        """


        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################

        #values, counts = np.unique(self.neighbors, return_counts=True, axis=0)
        #values[counts == np.max(counts),]
        #perc = np.array(counts/len(self.neighbors))
        #return values, perc

        #return np.random.randint(10, size=self.neighbors.size), np.random.random(self.neighbors.size)
        max_freq = []

        for i in self.neighbors:
            arr, freq = np.unique(i, return_counts=True)
            max_freq_type = arr[np.argmax(freq)]
            max_freq.append(max_freq_type)
        return max_freq


    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
        :param k:         :param k:  the number of neighbors to look at
        :return: the output form get_class (2 Nx1 vector, 1st the class 2nd the  % of votes it got
        """
        self.get_k_neighbours(test_data, k)
        return self.get_class()
