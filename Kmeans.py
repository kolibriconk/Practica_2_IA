__authors__ = ['1571610', '1562750', '1565479']
__group__ = 'DM.18'

import numpy as np
import utils
import math

class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options
        self._init_centroids()

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values

                if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                the last dimension
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension        """
        # Check that the values are float, if not convert it
        arr = np.array(X)
        if arr.dtype != "float64":
            arr = np.array(arr, dtype=np.float64)

        if len(arr.shape) == 3:
            n = arr.shape[0]*arr.shape[1]
            d = arr.shape[2]
        else:
            n = arr.shape[0]
            d = arr.shape[1]

        arr = arr.reshape(n, d)

        self.X = arr

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options == None:
            options = {}
        if not 'km_init' in options:
            options['km_init'] = 'first'
        if not 'verbose' in options:
            options['verbose'] = False
        if not 'tolerance' in options:
            options['tolerance'] = 0.10
        if not 'max_iter' in options:
            options['max_iter'] = np.inf
        if not 'fitting' in options:
            options['fitting'] = 'WCD'  # within class distance.

        self.options = options

    def _init_centroids(self):
        """
        Initialization of centroids
        """
        if self.options['km_init'].lower() == 'first':
            selected_pixels = {}
            self.centroids = []
            for i in range(self.K):
                for pixel in self.X:
                    aux = tuple(pixel)
                    if aux not in selected_pixels:
                        self.centroids.append(pixel)
                        selected_pixels[aux] = 1  # Marking the pixel as used so we don't repeat it
                        break
            self.centroids = np.copy(self.centroids)
            self.old_centroids = np.copy(self.centroids)

        elif self.options['km_init'].lower() == 'random':
            self.centroids = np.random.rand(self.K, self.X.shape[1])
            self.old_centroids = np.copy(self.centroids)

        elif self.options['km_init'].lower() == 'custom':
            self.centroids = [self.X[0]]
            for m in range(1, self.K):
                dist_sqr = np.array([min([np.inner(x-y, x-y) for x in self.centroids]) for y in self.X])
                prob = dist_sqr/dist_sqr.sum()
                cum_prob = prob.cumsum()
                r = np.random.rand()

                for k, n in enumerate(cum_prob):
                    if r < n:
                        i = k
                        break

                self.centroids.append(self.X[i])

            self.centroids = np.copy(self.centroids)
            self.old_centroids = np.copy(self.centroids)


    def get_labels(self):
        """        Calculates the closest centroid of all points in X
        and assigns each point to the closest centroid
        """
        self.labels = distance(self.X, self.centroids).argmin(axis=1)

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the
        """
        self.old_centroids = np.copy(self.centroids)
        iters = range(self.centroids.shape[0])
        for i in iters:
            if (self.labels == i).any():
                self.centroids[i] = self.X[(self.labels == i)].reshape(-1, self.X.shape[1]).mean(axis=0)


    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        return np.allclose(self.old_centroids, self.centroids, atol=self.options['tolerance'])

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number
        of iterations is smaller than the maximum number of iterations.
        """
        self._init_centroids()

        while self.num_iter < self.options['max_iter']:
            self.get_labels()
            self.get_centroids()
            self.num_iter += 1
            if self.converges():
                break

    def whitinClassDistance(self):
        """
         returns the whithin class distance of the current clustering
        """
        WCD = 0
        for c in range(self.centroids.shape[0]):
            if (self.labels == c).any():
                WCD += np.linalg.norm(self.X[self.labels == c] - self.centroids[c])**2
        for i, pixel in enumerate(self.X):
            aux = pixel - self.centroids[self.labels[i]]
            WCD += np.matmul(aux, aux.transpose())

        return WCD/self.X.shape[0]

    def find_bestK(self, max_K):
        """
         sets the best k anlysing the results up to 'max_K' clusters
        """

        tolerance = self.options['tolerance']
        last_dist = -1
        for k in range(2, max_K + 1):
            self.K = k
            self._init_centroids()
            self.num_iter = 0
            self.fit()
            res = self.whitinClassDistance()
            if last_dist != -1:
                aux = res / last_dist
                if (1 - aux) < tolerance:
                    self.K = k - 1
                    break

            last_dist = res
        return (1- aux)


def distance(X, C):
    """
    Calculates the distance between each pixcel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """
    distance_list = []
    for centroid in C:
        aux = X-centroid
        distance_list.append(np.linalg.norm(aux, axis=1))
    return np.transpose(distance_list)


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """
    color_probs = utils.get_color_prob(centroids)
    labels = []
    for i in range(centroids.shape[0]):
        color = utils.colors[color_probs[i].argmax()]
        labels.append(color)
    return labels
