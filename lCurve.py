import numpy as np

rbin = 3 #0 for none, 1-3 for deltar bins, 4 for deltar==0

#load data
print 'Loading data...'
bg = np.load('Octet_Rotated_withDR_new.npy')
sig = np.load('Singlet_Rotated_withDR_new.npy')

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
y = y[idx]

#training vs test
tr = int(0.8*X.shape[0])

print 'Loading keras...'

from keras import backend as K
K.set_image_dim_ordering('th')

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.layers import GlobalAveragePooling2D
from keras.optimizers import RMSprop, SGD
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, CSVLogger

dropout = 0.5 #0.7 (possbile larger size)
width = 1024
convWidth = 32 #32

print 'Building Network...'

model = Sequential()

model.add(Conv2D(convWidth, (3, 3), input_shape=(1, 32, 32), padding='same', activation='relu')) #kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPooling2D((2, 2), strides=2))

model.add(Conv2D(convWidth * 2, (3, 3), padding='same', activation='relu')) # kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPooling2D((2, 2), strides=2))

#add batch normalization?

model.add(Flatten())
#model.add(GlobalAveragePooling2D())

model.add(Dense(width))
model.add(Activation('relu'))
model.add(Dropout(dropout))

model.add(Dense(1))
model.add(Activation('sigmoid'))

#declare and compile model
sgd = SGD(lr=0.01, momentum=0, decay=0.0, nesterov=True)
model.compile(optimizer=sgd, loss='binary_crossentropy')
model.summary()

exp = np.zeros(10)
lear = np.zeros(10)
model.save_weights('model.h5')
for j in range(0,10):
    print j
    ntr = int(tr * float(j+1)/10)

    model.compile(optimizer=sgd, loss='binary_crossentropy') #, metrics=['binary_accuracy']
    model.load_weights('model.h5')

    h = model.fit(X[:ntr], y[:ntr],
    batch_size=256, 
    epochs=50,
    validation_split = 0.2)

    y_hat = model.predict(X[tr:])
    y_hat = y_hat.flatten()
    y_test = y[tr:]

    tpr = []
    fpr = []

    for i in range(1001):
        th = i/float(1000)
        TP = np.sum((y_hat[:] >= th) * y_test[:])
        tpr.append( TP / float(np.sum(y_test[:])) )

        FP = np.sum((y_hat[:] >= th) * (1-y_test[:]))
        fpr.append( FP / float(np.sum(1-y_test[:])) )

    tpr = np.sort(np.concatenate([[0.0], tpr]))
    fpr = np.sort(np.concatenate([[0.0], fpr]))
    AUC = np.trapz(tpr, x=fpr)

    print int(0.8*ntr)
    print AUC

    exp[j] = int(0.8*ntr)
    lear[j] = AUC

print exp
print lear