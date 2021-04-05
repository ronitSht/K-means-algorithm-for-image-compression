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


# data preperation (loading, normalizing, reshaping)
path = 'dog.jpeg'

A = imread(path)
A_norm = A.astype(float) / 255.
img_size = A_norm.shape
# arr of pixels - use the x
X = A_norm.reshape(img_size[0] * img_size[1], img_size[2])

# struct(){
#   count = 0;
# x1 = 0;
# x2 = 0;
# x3 = 0;
# };

# main()

# K=2
# while (K <= 16):
arrZToPrint = [None] * 2
arrZ = init_centroids(X, 2)
for p in range(0, len(arrZ)):
    arrZToPrint[p] = numpy.floor(arrZ[p] * 100) / 100
strList = ', '.join(str(j) for j in arrZToPrint)
print("iter 0: " + strList)

# def distance(X,K):
# runs through the points z
distArrOfEachPointToZ = [len(X)]
minDistArr = []
arrOfClosestCentroidForEachPoint = [None] * 128
counter = 0
indexOfMin = 0
# len X = 128
print(X)
for p in range(0,128):
    print(X[p])

dist = distance.euclidean([0.04705882, 0.01960784, 0], arrZ[0])
print("yyyyyyyy")
print(dist)
dists = distance.euclidean([0.04705882, 0.01960784, 0], arrZ[1])
print(dists)

for e in range(0, 1):
    for i in range(0, 128):
        # number of z
        dist = distance.euclidean(X[i], arrZ[0])
        print("dst:")
        print(dist)
        # dist = numpy.linalg.norm(X[i] - arrZ[0])
        for j in range(1, 2):
            # if (K == 2):
            newDist = distance.euclidean(X[i], arrZ[j])
            print("new:")
            print(newDist)
            # newDist = numpy.linalg.norm(X[i] - arrZ[j])
            if dist > newDist:
                dist = newDist
                indexOfMin = 1
                arrOfClosestCentroidForEachPoint[i] = indexOfMin
            else:
                indexOfMin = 0
                arrOfClosestCentroidForEachPoint[i] = indexOfMin
        # arrOfClosestCentroidForEachPoint.append(indexOfMin)
    print(arrOfClosestCentroidForEachPoint)

    # def avg():
    avg = 0
    sumOfPoints = 0
    countNumOfPointsInCluster = 0
    # size K -2/4/8/16
    avgArrNewCent = [None] * 2
    # 128 - len(arrOfClosestCentroidForEachPoint)


    # arrCentAndPoints = [2][128][3]
    # arrCentAndPoints = [[[[0] * 3] * 128] * 2]
    # runs through the centroids
    for m in range(0, 2):
        for n in range(0, 128):
            # m is the index of the centroid
            if arrOfClosestCentroidForEachPoint[n] == m:
                print("bye")
                # put the value of the point
                #  arrCentAndPoints[m][n] = X[n]
                countNumOfPointsInCluster += 1
                # print(countNumOfPointsInCluster)
                sumOfPoints += numpy.array(X[n])
                # print(sumOfPoints)

        if countNumOfPointsInCluster != 0:
            print("ddddddd")
            avg = sumOfPoints / countNumOfPointsInCluster
            avg = numpy.floor(avg * 100) / 100
            # print(avg)
            avgArrNewCent[m] = avg
            # avgArrNewCent.append(avg)
        # equal to 0 - there are no points that are closest to the centroid, so it remains the same the centroid.
        else:
            print("hi")
            arrZ[m] = numpy.floor(arrZ[m] * 100) / 100
            avgArrNewCent[m] = arrZ[m]
            # avgArrNewCent.append(arrZ[m])
            # print(arrZ)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    strList = ', '.join(str(j) for j in avgArrNewCent)
    indexToPrint = e + 1
    print("iter " + str(indexToPrint) + ": " + strList)


# K*=2







# increase the K to 2/4/8/16 - while loop in the end

# avg X  - pixels - 10
# for i in range(10):




# float


# distance between k and z
# avg each z
# 10 times
