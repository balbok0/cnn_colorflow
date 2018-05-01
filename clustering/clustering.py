import numpy as np

algorithm = 2 # 0 for k-mean, 1 for fuzzy k-means (doesnt exist in scikit), 
              # 2 for Hierarchical clustering, 3 for Gaussian mixtures (too long time to run)
nClusters = 20


#load data
print 'Loading data...'
bg = np.load('../Octet_Rotated_withDR_new.npy')
sig = np.load('../Singlet_Rotated_withDR_new.npy')

#remove nans
mask = ~np.isnan(bg).any(axis=1)
bg = bg[mask[:],...]
mask = ~np.isnan(sig).any(axis=1)
sig = sig[mask[:],...]

#get image
sig = sig[:, 9:-1]
bg = bg[:, 9:-1]

#make truth values for later analysis
y = np.zeros(bg.shape[0] + sig.shape[0])
y[bg.shape[0]:] = 1

#append bg and sig
X = np.concatenate((bg, sig), axis = 0)

print 'Training...'
from sklearn.cluster import KMeans, FeatureAgglomeration
from sklearn.mixture import GaussianMixture

'''
#good classifier?
cluster = KMeans(n_clusters=2).fit(X)
print np.sum(cluster.labels_ == y)/float(y.shape[0])

'''
if algorithm == 0:
    cluster = KMeans(n_clusters=nClusters).fit(X)
    title = 'KMeans'
    prediction = cluster.labels_
elif algorithm == 2:
    cluster = FeatureAgglomeration(n_clusters=nClusters).fit(X)
    title = 'HierarchicalClustering'
    prediction = cluster.predict(X)
elif algorithm == 3:
    cluster = GaussianMixture(n_components=nClusters).fit(X)
    title = 'GaussianMixtures'
    prediction = cluster.predict(X)

print 'Generating plots...'
from matplotlib import pyplot as plt

for i in range(0, nClusters):
    mask = (prediction == i)
    xSample = X[mask]

    sigCount = np.sum(y[mask])
    bgCount = np.sum(1 - y[mask])

    plt.clf()
    plt.imshow(np.sum(xSample, axis=0).reshape(25, 25), interpolation="none", cmap='GnBu')
    plt.xlabel('Proportional to Translated Pseudorapidity', fontsize=7)
    plt.ylabel('Proportional to Translated Azimuthal Angle', fontsize=7)
    plt.title('Cluster ' + str(i) + ', #oct=' + str(bgCount) + ', #sing=' + str(sigCount), fontsize=14)
    plt.colorbar()
    plt.savefig(title + '/cluster_' + str(i) + '.png')

print 'Done'
