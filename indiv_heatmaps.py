import numpy as np

rbin = 0 #0 for none, 1-3 for deltar bins, 4 for deltar==0
weighting = 1 #1 if weighting mass

#load data
print 'Loading data...'
bg = np.load('Octet_Rotated_withDR_new.npy')
sig = np.load('Singlet_Rotated_withDR_new.npy')

bg = bg[:75000]
sig = sig[:75000]

#remove nans
mask = ~np.isnan(bg).any(axis=1)
bg = bg[mask[:],...]
mask = ~np.isnan(sig).any(axis=1)
sig = sig[mask[:],...]

#bin to deltar
if rbin == 1:
    bg = bg[np.where((np.absolute(bg[:, -1] - .25) < .25))]
    sig = sig[np.where((np.absolute(sig[:, -1] - .25) < .25))]
if rbin == 2:
    bg = bg[np.where((np.absolute(bg[:, -1] - .6) <= .1))]
    sig = sig[np.where((np.absolute(sig[:, -1] - .6) <= .1))]
if rbin == 3:
    bg = bg[np.where(bg[:, -1] - .7 > 0)]
    sig = sig[np.where(sig[:, -1] - .7 > 0)]
if rbin == 4:
    bg = bg[np.where(bg[:, -1] == 0)]
    sig = sig[np.where(sig[:, -1] == 0)]

#weighting
sweights = np.ones(sig.shape[0])
if weighting == 1:
    mxmin = 0
    mxmax = 400
    mymin = 1
    mymax = 100000
    mbins = np.linspace(mxmin, mxmax, 100)

    for i in range(1, 100):
        locb = (bg[:, 6] < mbins[i]) & (bg[:, 6] >= mbins[i-1])
        locs = (sig[:, 6] < mbins[i]) & (sig[:, 6] >= mbins[i-1])
        n = bg[locb].shape[0]
        d = sig[locs].shape[0]
        if n==0 or d==0:
            sweights[locs] = 0
        else:
            sweights[locs] = float(n)/float(d)

bweights = np.ones(bg.shape[0])
Xweights = np.concatenate((bweights, sweights), axis = 0)

#reshape
bg = bg[:, 9:-1].reshape(bg.shape[0], 1, 25, 25)
cvrt = bg.shape[0]
sig = sig[:, 9:-1].reshape(sig.shape[0], 1, 25, 25)
X = np.concatenate((bg, sig), axis = 0)
y = np.zeros(X.shape[0])
y[cvrt:]=1

#ZERO PAD TO 32x32
X = np.lib.pad(X, ((0, 0), (0, 0), (0, 7), (0, 7)), 'constant', constant_values=0)

#shuffle
idx = range(X.shape[0])
np.random.shuffle(idx)
X = X[idx]
Xweights = Xweights[idx]
y = y[idx]

#train vs test
tr = int(0.8*X.shape[0])
X_train = X[:tr]
X_test = X[tr:]

Xweights = Xweights[:tr]

y_train = y[:tr]
y_test = y[tr:]

#load in keras
print 'Loading keras...'

from keras import backend as K
K.set_image_dim_ordering('th')

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.layers import GlobalAveragePooling2D
from keras.optimizers import RMSprop
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, CSVLogger

dropout = 0.5 #0.7 (possbile larger size)
width = 125 #625
convWidth = 16 #32

print 'Building Network...'

model = Sequential()

model.add(Conv2D(convWidth, (11, 11), input_shape=(1, 32, 32), padding='same')) #padding='same', kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(convWidth * 2, (3, 3), padding='same')) # padding='same', kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

#add batch normalization?

model.add(Flatten())
#model.add(GlobalAveragePooling2D())

model.add(Dense(width))
model.add(Activation('relu'))
model.add(Dropout(dropout))

model.add(Dense(width))
model.add(Activation('relu'))
model.add(Dropout(dropout))

model.add(Dense(1))
model.add(Activation('sigmoid'))

#declare and compile model
model.compile(optimizer = RMSprop(lr=0.0001), loss = 'binary_crossentropy', metrics=['mae'])
model.summary()

#call network and train 
print ('Training:')

if weighting != 1:
    Xweights = None

h = model.fit(X_train, y_train,
    batch_size=1024, 
    epochs=200,
    validation_split = 0.2,
    callbacks = [
        ModelCheckpoint('best_model_cnn.hdf5', monitor='val_loss', verbose=2, save_best_only=True)
        #CSVLogger('cnn_training.log')
    ],
    sample_weight=Xweights)

print 'Loading Best Model'
h = load_model('best_model_cnn.hdf5')

#generating heat maps
print 'Generating Heat Maps'
from matplotlib import pyplot as plt

#for average heatmaps later
acc_oct = np.zeros((25, 25))
acc_sing = np.zeros((25, 25))

sample_n = 1000
for k in range(sample_n):
    print 'PROGRESS: '+ str(k) + ' / 1000 images calculated'
    acc = np.zeros((29, 29))
    for j in range(625):

        print 'progress: '+ str(j) + ' / 625 pixels calculated'

        ver = j % 25
        hor = int(j // 25)

        #depad and wrap
        X_test_b = np.pad(X_test[k:k+1, :, :25, :25], ((0, 0), (0, 0), (2, 2), (2, 2)), mode='wrap')

        #block 5 by 5 squares, starting in upper left corner of pad
        #(the first block will center on the first real pixel, not pad)
        X_test_b[:, 0, hor:(hor+5), ver:(ver+5)] = 0

        #de-wrap and re-pad
        X_test_b = np.lib.pad(X_test_b[:, :, 2:27, 2:27], ((0, 0), (0, 0), (0, 7), (0, 7)), 'constant', constant_values=0)

        #how good is prediction?
        prediction = np.array(model.predict(X_test_b))[0][0]
        if y_test[k] == 0:
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

    type = ''
    if y_test[k] == 0:
        type = 'octet'
        acc_oct += acc
    else:
        acc_sing += acc
        type = 'singlet'

    #plot heatmap
    plt.clf()
    plt.imshow(AUC, interpolation="none", cmap='GnBu', vmin=0, vmax=25)
    plt.xlabel('Proportional to Translated Pseudorapidity', fontsize=7)
    plt.ylabel('Proportional to Translated Azimuthal Angle', fontsize=7)
    plt.title('Heatmap' + '-' + type, fontsize=19)
    plt.colorbar()
    plt.savefig('heatmap/indivs_low_res/' + str(k) + '.png')

    #plot image
    plt.clf()
    plt.imshow(X_test[k, 0, :25, :25], interpolation="none", cmap='GnBu')
    plt.xlabel('Proportional to Translated Pseudorapidity', fontsize=7)
    plt.ylabel('Proportional to Translated Azimuthal Angle', fontsize=7)
    plt.title('Image' + '-' + type, fontsize=19)
    plt.colorbar()
    plt.savefig('heatmap/images/' + str(k) + '.png')

#plot octet
plt.clf()
plt.imshow(acc_oct, interpolation="none", cmap='GnBu', vmin=0, vmax=15*sample_n) #15 determined by trial
plt.xlabel('Proportional to Translated Pseudorapidity', fontsize=7)
plt.ylabel('Proportional to Translated Azimuthal Angle', fontsize=7)
plt.title('Octet Average', fontsize=19)
plt.colorbar()
plt.savefig('heatmap/octet.png')

#plot singlet
plt.clf()
plt.imshow(acc_sing, interpolation="none", cmap='GnBu', vmin=0, vmax=15*sample_n)
plt.xlabel('Proportional to Translated Pseudorapidity', fontsize=7)
plt.ylabel('Proportional to Translated Azimuthal Angle', fontsize=7)
plt.title('Singlet Average', fontsize=19)
plt.colorbar()
plt.savefig('heatmap/singlet.png')

print 'Done!'