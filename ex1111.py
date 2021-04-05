import numpy
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.misc import imread


def init_centroids(X, K):
    """
    Initializes K centroids that are to be used in K-Means on the dataset X.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Samples, where n_samples is the number of samples and n_features is the number of features.
    K : int
        The number of centroids.

    Returns
    -------
    centroids : ndarray, shape (K, n_features)
    """
    if K == 2:
        return np.asarray([[0., 0., 0.],
                           [0.07843137, 0.06666667, 0.09411765]])
    elif K == 4:
        return np.asarray([[0.72156863, 0.64313725, 0.54901961],
                           [0.49019608, 0.41960784, 0.33333333],
                           [0.02745098, 0., 0.],
                           [0.17254902, 0.16862745, 0.18823529]])
    elif K == 8:
        return np.asarray([[0.01568627, 0.01176471, 0.03529412],
                           [0.14509804, 0.12156863, 0.12941176],
                           [0.4745098, 0.40784314, 0.32941176],
                           [0.00784314, 0.00392157, 0.02745098],
                           [0.50588235, 0.43529412, 0.34117647],
                           [0.09411765, 0.09019608, 0.11372549],
                           [0.54509804, 0.45882353, 0.36470588],
                           [0.44705882, 0.37647059, 0.29019608]])
    elif K == 16:
        return np.asarray([[0.61568627, 0.56078431, 0.45882353],
                           [0.4745098, 0.38039216, 0.33333333],
                           [0.65882353, 0.57647059, 0.49411765],
                           [0.08235294, 0.07843137, 0.10196078],
                           [0.06666667, 0.03529412, 0.02352941],
                           [0.08235294, 0.07843137, 0.09803922],
                           [0.0745098, 0.07058824, 0.09411765],
                           [0.01960784, 0.01960784, 0.02745098],
                           [0.00784314, 0.00784314, 0.01568627],
                           [0.8627451, 0.78039216, 0.69803922],
                           [0.60784314, 0.52156863, 0.42745098],
                           [0.01960784, 0.01176471, 0.02352941],
                           [0.78431373, 0.69803922, 0.60392157],
                           [0.30196078, 0.21568627, 0.1254902],
                           [0.30588235, 0.2627451, 0.24705882],
                           [0.65490196, 0.61176471, 0.50196078]])
    else:
        print('This value of K is not supported.')
        return None


# the print function
def print_cent(cent):
    if type(cent) == list:
        cent = np.asarray(cent)
    if len(cent.shape) == 1:
        return ' '.join(str(np.floor(100 * cent) / 100).split()).replace('[ ', '[').replace('\n', ' ').replace(' ]',
                                                                                                               ']').replace(
            ' ', ', ')
    else:
        return ' '.join(str(np.floor(100 * cent) / 100).split()).replace('[ ', '[').replace('\n', ' ').replace(' ]',
                                                                                                               ']').replace(
            ' ', ', ')[1:-1]


def main():
    # data preperation (loading, normalizing, reshaping)
    path = 'dog.jpeg'

    A = imread(path)
    A_norm = A.astype(float) / 255.
    img_size = A_norm.shape
    # arr of pixels - use the x
    X = A_norm.reshape(img_size[0] * img_size[1], img_size[2])

    K = 2
    while (K <= 16):
        print("k=" + str(K)+":")
        arrZToPrint = [None] * K
        # puts in arrZ the initial centroids, and also the new centroids will be put in arrZ.
        arrZ = init_centroids(X, K)

        # prints the first iteration
        for p in range(0, len(arrZ)):
            arrZToPrint[p] = arrZ[p]
        lisNP = np.array(arrZToPrint)
        print("iter 0:", print_cent(lisNP))

        # def distance(X,K):
        # runs through the points z
        distArrOfEachPointToZ = [len(X)]
        minDistArr = []
        arrOfClosestCentroidForEachPoint = [None] * len(X)
        counter = 0
        # len X = 128
        # the ten iterations
        for e in range(0, 10):
            # runs through the points (pixels) in X and finds to each point the closest centroid (in arrZ).
            for i in range(0, len(X)):
                # number of z
                dist = distance.euclidean(X[i], arrZ[0])
                indexOfMin = 0
                for j in range(1, K):
                    newDist = distance.euclidean(X[i], arrZ[j])
                    # if newDist is the minimal distance, we keep the index of the centroid.
                    if dist > newDist:
                        dist = newDist
                        indexOfMin = j
                # puts in this array the index of the closest centoid of each of the 128 points in X
                arrOfClosestCentroidForEachPoint[i] = indexOfMin

            # def avg():
            avgArrNewCent = [None] * K
            # 128 - len(arrOfClosestCentroidForEachPoint)
            # runs through the centroids
            for m in range(0, K):
                avg = 0
                sumOfPoints = 0
                countNumOfPointsInCluster = 0
                for n in range(0, len(X)):
                    # m is the index of the centroid
                    if arrOfClosestCentroidForEachPoint[n] == m:
                        countNumOfPointsInCluster += 1
                        sumOfPoints += numpy.array(X[n])

                # if there are centroids that have points that are in their cluster (they are closest to them),
                # it calculates the average (the new centroid of the cluster.
                if countNumOfPointsInCluster != 0:
                    avg = sumOfPoints / countNumOfPointsInCluster
                    arrZ[m] = avg
                    avgArrNewCent[m] = avg
                # equal to 0 - there are no points that are closest to the centroid, so it remains the same centroid.
                else:
                    avgArrNewCent[m] = arrZ[m]

            # prints the array of the new centroids.
            listNP = np.array(avgArrNewCent)
            indexToPrint = e + 1
            print("iter {}: {}".format(indexToPrint, print_cent(listNP)))

        K *= 2

if __name__ == "__main__":
    main()
