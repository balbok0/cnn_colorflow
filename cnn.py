import numpy as np

rbin = 0 #0 for none, 1-3 for deltar bins, 4 for deltar==0
weighting = 0 #1 if weighting mass

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

## TODO: everything above this could be new data preprocessing script

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
model.compile(optimizer = RMSprop(lr=0.0003), loss = 'binary_crossentropy', metrics=['mae'])
model.summary()

# TODO: Everything above this could be a model creation script

#training vs test
tr = int(0.8*X.shape[0])

#call network and train 
print ('Training:')

if weighting != 1:
    Xweights = None
else:
    Xweights = Xweights[:tr]

h = model.fit(X[:tr], y[:tr],
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

#TODO : Everything above this, model training script

#use new model to predict
print 'Predicting'

y_hat = model.predict(X[tr:])
y_hat = y_hat.flatten()

#print("y_hat.shape", y_hat.shape)
#print("y_hat", y_hat[0:10])

#create and save out ROC curves
print 'Creating ROC curves'

y_test = y[tr:]

tpr = []
fpr = []

for i in range(100001):
    th = i/float(100000)
    TP = np.sum((y_hat[:] >= th) * y_test[:])
    tpr.append( TP / float(np.sum(y_test[:])) )

    FP = np.sum((y_hat[:] >= th) * (1-y_test[:]))
    fpr.append( FP / float(np.sum(1-y_test[:])) )

tpr = np.concatenate([[0.0], tpr])
fpr = np.concatenate([[0.0], fpr])

np.savetxt("CSVs/tpr.csv", np.sort(tpr), delimiter=',')
np.savetxt("CSVs/fpr.csv", np.sort(fpr), delimiter=',')

#check for overfitting, by looking at test ROC
#print 'Checking overfit'

#y_hat = model.predict(X[:tr])
#y_hat = y_hat.flatten()

#y_test = y[:tr]

#tpr = []
#fpr = []

#for i in range(100001):
#    th = i/float(100000)
#    TP = np.sum((y_hat[:] >= th) * y_test[:])
#    tpr.append( TP / float(np.sum(y_test[:])) )

#    FP = np.sum((y_hat[:] >= th) * (1-y_test[:]))
#    fpr.append( FP / float(np.sum(y_test[:])) )

#tpr = np.concatenate([[0.0], tpr])
#fpr = np.concatenate([[0.0], fpr])

#np.savetxt("CSVs/tpr_o.csv", np.sort(tpr), delimiter=',')
#np.savetxt("CSVs/fpr_o.csv", np.sort(fpr), delimiter=',')

print 'Done!'