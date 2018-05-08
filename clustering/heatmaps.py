import numpy as np

#load in keras
print 'Loading keras...'

from keras import backend as K
K.set_image_dim_ordering('th')

rbin = 0 #0 for none, 1-3 for deltar bins, 4 for deltar==0
weighting = 1 #1 if weighting mass

#load data
print 'Loading data...'
bg = np.load('../Octet_Rotated_withDR_new.npy')
sig = np.load('../Singlet_Rotated_withDR_new.npy')

bg = bg[:75000]
sig = sig[:75000]

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
X = X.reshape(X.shape[0], 1, 25, 25)

print 'Loading Best Model'
from keras.models import load_model
model = load_model('best_model_cnn.hdf5')

#generating heat maps
print 'Generating Heat Maps'
from matplotlib import pyplot as plt

clusters = np.load('clusters.npy')

from math import log

sample_n = 20
for k in range(10, sample_n):
    print 'PROGRESS: '+ str(k) + ' / ' + str(sample_n) + ' clusters calculated'
    acc_total = np.zeros((25, 25))
    mask = (clusters == k)
    xSample = X[mask]
    ran = min(xSample.shape[0], 1000)
    for i in range(ran):
        acc = np.zeros((29, 29))

        for j in range(625):

            #print 'progress: '+ str(j) + ' / 625 pixels calculated'

            #format and wrap
            X_test_b = np.pad(xSample[i:(i+1), :, :, :], ((0, 0), (0, 0), (2, 2), (2, 2)), mode='wrap')

            ver = j % 25
            hor = int(j // 25)

            #block 5 by 5 squares, starting in upper left corner of pad
            #(the first block will center on the first real pixel, not pad)
            for m in range(hor, hor + 5):
                for n in range(ver, ver + 5):
                    if m != 14 or n != 14: 
                        X_test_b[:, 0, m, n] = 0

            #de-wrap and pad
            X_test_b = np.lib.pad(X_test_b[:, :, 2:27, 2:27], ((0, 0), (0, 0), (0, 7), (0, 7)), 'constant', constant_values=0)

            #how good is prediction?
            prediction = np.array(model.predict(X_test_b))[0][0]
            if y[np.asarray(np.nonzero(mask))[0, i]] == 0:
                prediction = 1 - prediction
            
            acc[hor:(hor+5), ver:(ver+5)] += prediction

        acc = acc[2:27, 2:27].reshape(25, 25)

        #edges need normalized
        acc[0, :] *= 5.0/3.0 #normal passes / edge passes
        acc[:, 0] *= 5.0/3.0
        acc[24, :] *= 5.0/3.0
        acc[:, 24] *= 5.0/3.0
        acc[1, :] *= 5.0/4.0
        acc[:, 1] *= 5.0/4.0
        acc[23, :] *= 5.0/4.0
        acc[:, 23] *= 5.0/4.0

        acc_total += acc

        if i < 3:
            #plot heatmap
            plt.clf()
            plt.imshow(np.log(acc), interpolation="none", cmap='jet', vmin=2.2, vmax=3) # experimentall chosen
            plt.xlabel('Proportional to Translated Pseudorapidity', fontsize=7)
            plt.ylabel('Proportional to Translated Azimuthal Angle', fontsize=7)
            plt.title('Heatmap', fontsize=19)
            plt.colorbar()
            plt.savefig('KMeans_Seperated/cluster_' + str(k) + '/' + str(i) + '_heat.png')

    #plot average
    plt.clf()
    plt.imshow(np.log(acc_total), interpolation="none", cmap='jet', vmin=log(9 * ran), vmax=log(20 * ran))
    plt.xlabel('Proportional to Translated Pseudorapidity', fontsize=7)
    plt.ylabel('Proportional to Translated Azimuthal Angle', fontsize=7)
    plt.title('Average Heatmap for cluster', fontsize=19)
    plt.colorbar()
    plt.savefig('KMeans_Seperated/cluster_' + str(k) + '/average_1000_heat.png')

    print '\n'

print 'Done!'