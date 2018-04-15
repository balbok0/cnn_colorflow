import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib import pyplot as plt
from scipy.stats.mstats import spearmanr

flag = 1
p2modifier = -750
scat = 0
spearman = 1

#load data
print 'Loading data...'
bg = np.load('Octet_Rotated_withDR_new.npy')
sig = np.load('Singlet_Rotated_withDR_new.npy')

#remove nans
mask = ~np.isnan(bg).any(axis=1)
bg = bg[mask[:],...]
mask = ~np.isnan(sig).any(axis=1)
sig = sig[mask[:],...]

#if flagged, use only delta_r!=0
if flag == 1:
    bg = bg[np.where(bg[:, -1] != 0)]
    sig = sig[np.where(sig[:, -1] != 0)]

#make my metas
bgim = bg[:, 9:-1].reshape(bg.shape[0], 25, 25)
sigim = sig[:, 9:-1].reshape(sig.shape[0], 25, 25)

def mymetab (rangexs, rangexe, rangeys, rangeye):
    return np.nansum(np.nansum(bgim[:, rangeys:rangeye, rangexs:rangexe], axis = 1), axis = 1)

def mymetas (rangexs, rangexe, rangeys, rangeye):
    return np.nansum(np.nansum(sigim[:, rangeys:rangeye, rangexs:rangexe], axis = 1), axis = 1)

#my constants
y_1 = 11
y_2 = 14
y_3 = 19
y_4 = 25

#first way : 11,14 10,15 9,16
#second way : 11,14 5,20 5,20

bgmy1 = mymetab(11, 14, y_1, y_2) #primary subjet
bgmy2 = mymetab(5, 20, y_2, y_3) #middle
bgmy3 = mymetab(5, 20, y_3, y_4) #secondary subjet

sigmy1 = mymetas(11, 14, y_1, y_2) #primary subjet
sigmy2 = mymetas(5, 20, y_2, y_3) #middle
sigmy3 = mymetas(5, 20, y_3, y_4) #secondary subjet

#metavariables
bgp1  = bg[:, 0]
bgp2 = bg[:, 1]
bgm = bg[:, 6]
bgr = bg[:, -1]

sigp1  = sig[:, 0]
sigp2 = sig[:, 1]
sigm = sig[:, 6]
sigr = sig[:, -1]

print 'Predicting'

#make reality
y = np.zeros(bg.shape[0] + sig.shape[0])
y[bg.shape[0]:]=1

X = np.concatenate((bg[:, 9:-1].reshape(bg.shape[0], 625), sig[:, 9:-1].reshape(sig.shape[0], 625)), axis = 0)

#make predictions
def predict(bg, sig, y):
    X = np.concatenate((bg, sig)).reshape(-1, 1)
    clf = LinearDiscriminantAnalysis(solver='svd')
    clf.fit(X, y)
    return clf.predict(X)

y_hat_p1 = predict(bgp1, sigp1, y)
y_hat_p2 = predict(bgp2, sigp2[:p2modifier], y[:p2modifier])
y_hat_m = predict(bgm, sigm, y)
y_hat_r = predict(bgr, sigr, y)

#use all
X_all = np.zeros((bgp1.shape[0]+sigp1.shape[0], 4))
X_all[:, 0] =  np.concatenate((bgp1, sigp1))
X_all[:, 1] =  np.concatenate((bgp2, sigp2))
X_all[:, 2] =  np.concatenate((bgm, sigm))
X_all[:, 3] =  np.concatenate((bgr, sigr))
clf = LinearDiscriminantAnalysis(solver='svd')
clf.fit(X_all, y)
y_hat_all = clf.predict(X_all)

#use mymeta
X_my = np.zeros((bgp1.shape[0]+sigp1.shape[0], 3))
X_my[:, 0] =  np.concatenate((bgmy1, sigmy1))
X_my[:, 1] =  np.concatenate((bgmy2, sigmy2))
X_my[:, 2] =  np.concatenate((bgmy3, sigmy3))
clf = LinearDiscriminantAnalysis(solver='svd')
clf.fit(X_my, y)
y_hat_my = clf.predict(X_my)

#use delta r and m
X_rm = np.zeros((bgp1.shape[0]+sigp1.shape[0], 2))
X_rm[:, 0] =  np.concatenate((bgm, sigm))
X_rm[:, 1] =  np.concatenate((bgr, sigr))
clf = LinearDiscriminantAnalysis(solver='svd')
clf.fit(X_rm, y)
y_hat_rm = clf.predict(X_rm)

#use pulls
X_pull = np.zeros((bgp1.shape[0]+sigp1.shape[0], 2))
X_pull[:, 0] =  np.concatenate((bgp1, sigp1))
X_pull[:, 1] =  np.concatenate((bgp2, sigp2))

#make images
print 'Creating PCC Images'

def pearson(X, y_hat, title):
    y_pearson = np.zeros(X.shape)
    for i in range(X.shape[0]):
        y_pearson[i, :] = np.full(X.shape[1], y_hat[i])

    x_net = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        if spearman == 1:
            x_pearson = spearmanr(X[:, i], y_pearson[:, i]) #, rowvar = False
            x_net[i] = x_pearson[0]
        else:
            x_pearson = np.corrcoef(X[:, i], y_pearson[:, i])
            x_net[i] = x_pearson[0, 1]
    x_image = x_net.reshape(25, 25)

    plt.clf()
    plt.imshow(x_image, interpolation="none", cmap='seismic', vmin=-0.2, vmax=0.2)
    plt.xlabel('Proportional to Translated Pseudorapidity', fontsize=10)
    plt.ylabel('Proportional to Translated Azimuthal Angle', fontsize=10)
    plt.title('PCC for ' + title + ' and pixel intensity', fontsize=15)
    plt.colorbar()
    if spearman == 1:
        plt.savefig('pcc/Spearman_'  + title + '.png')
    else:
        plt.savefig('pcc/Pearson_'  + title + '.png')

    return x_net

X_p1_pearson = pearson(X, y_hat_p1, 'pull1')
X_p2_pearson = pearson(X[:p2modifier], y_hat_p2, 'pull2')
X_m_pearson = pearson(X, y_hat_m, 'mass')
X_r_pearson = pearson(X, y_hat_r, 'delta_r')
pearson(X, y, 'true')
pearson(X, y_hat_all, 'all')
pearson(X, y_hat_rm, 'delta_r_mass')
pearson(X, y_hat_my, 'my_meta')

#make scatter plots
if scat == 1:
    print 'Scattering'
    def scatter(y, X, variable):
        for i in range(y.shape[1]):
            plt.clf()
            plt.scatter(X[:], y[:, i])
            plt.xlabel(variable, fontsize=10)
            plt.ylabel('Pixel Intensity', fontsize=10)
            plt.title('Relationship between ' + variable + ' and pixel ' + str(i), fontsize=15)
            plt.savefig('pcc/scatters/' + variable + '/'  + str(i) + '.png')

    scatter(X, y_hat_p1, 'pull1')
    scatter(X[:p2modifier], y_hat_p2, 'pull2')
    scatter(X, y_hat_m, 'mass')
    scatter(X, y_hat_r, 'delta_r')

#ROC y vs y_hat_rm
X_r = np.concatenate((bgr, sigr)).reshape(-1, 1)
clf = LinearDiscriminantAnalysis(solver='svd')
clf.fit(X_my, y)
y_hat = clf.predict_proba(X_my)[:, 1]

y_test = y

tpr = []
fpr = []

for i in range(1001):
    th = i/float(1000)
    TP = np.sum((y_hat[:] >= th) * y_test[:])
    tpr.append( TP / float(np.sum(y_test[:])) )

    FP = np.sum((y_hat[:] >= th) * (1-y_test[:]))
    fpr.append( FP / float(np.sum(1-y_test[:])) )

tpr = np.concatenate([[0.0], tpr])
fpr = np.concatenate([[0.0], fpr])

np.savetxt("CSVs/tpr.csv", np.sort(tpr), delimiter=',')
np.savetxt("CSVs/fpr.csv", np.sort(fpr), delimiter=',')

print 'Done!'