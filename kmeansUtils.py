# -*- coding: utf-8 -*-
from sklearn.cluster import KMeans
from utils.STRTools import str_groupBy, str_parallel
import numpy as np
from scipy.spatial.distance import cdist

def computeKMeans(image, n_clusters=5):
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    clt = KMeans(n_clusters = n_clusters)
    imReshaped = image.reshape((image.shape[0] * image.shape[1], 3))
    clt.fit(imReshaped)
    
    groups = str_groupBy(clt.labels_, key=lambda x: x)
    hist = [len(groups[i])/float(clt.labels_.size) for i in range(3)]
    return clt.cluster_centers_.astype(np.uint8), hist

def assignColor(x, clusters):
    return clusters[np.argmin(cdist([x],clusters))]

def rerenderKMeans(image, clusters):
    imReshaped = image.reshape((image.shape[0] * image.shape[1], 3))
    new_image = np.array(str_parallel(assignColor, nThreads=8, chunksize=5000, pbar=False)(imReshaped, clusters=clusters))
    new_image_reshaped = new_image.reshape((image.shape[0], image.shape[1], 3))
    return new_image_reshaped