__authors__ = ['1571610', '1562750', '1565479']
__group__ = 'DM.18'

import numpy as np
import Kmeans
import KNN
from utils_data import read_dataset, visualize_k_means, visualize_retrieval
import matplotlib.pyplot as plt
import cv2

if __name__ == '__main__':

    #Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, \
    test_imgs, test_class_labels, test_color_labels = read_dataset(ROOT_FOLDER='./images/', gt_json='./test/gt.json')

    #List with all the existant classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))
    e = Kmeans_stadistics(Kmeans, 20)




## You can start coding your functions here

def Kmean_stadistics(Kmeans, Kmax):
    Kmeans.__init__(test_imgs)
    for k in Kmax:
        if k > 2:
            Kmeans.fit()
            Kmeans.whitinClassDistance()
    
    revisualize_k_means(Kmeans, test_imgs.shape())










