import numpy as np

rbin = 0 #0 for none, 1-3 for deltar bins, 4 for deltar==0
weighting = 0 #1 if weighting mass

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
cvrts = sig.shape[0]
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

tr = int(0.8*X.shape[0])

#load in keras
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
width = 1024 #625
convWidth = 32 #32

print 'Building Network...'

model = Sequential()

model.add(Conv2D(convWidth, (11, 11), input_shape=(1, 32, 32), padding='same')) #kernel_regularizer=regularizers.l2(0.01)))
layer1 = Activation('relu')
model.add(layer1)
model.add(MaxPooling2D((2, 2), strides=2))

model.add(Conv2D(convWidth * 2, (3, 3), padding='same')) # kernel_regularizer=regularizers.l2(0.01)))
layer2 = Activation('relu')
model.add(layer2)
model.add(MaxPooling2D((2, 2), strides=2))

#add batch normalization?

model.add(Flatten())
#model.add(GlobalAveragePooling2D())

model.add(Dense(width))
layer3 = Activation('relu')
model.add(layer3)
model.add(Dropout(dropout))

model.add(Dense(1))
model.add(Activation('sigmoid'))

#declare and compile model
sgd = SGD(lr=0.01, momentum=0, decay=0.0, nesterov=True)
model.compile(optimizer=sgd, loss='binary_crossentropy')
model.summary()

layer_1_vis = K.function([K.learning_phase()]+model.inputs, [layer1.output])
layer_2_vis = K.function([K.learning_phase()]+model.inputs, [layer2.output])
layer_3_vis = K.function([K.learning_phase()]+model.inputs, [layer3.output])

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
    ],
    sample_weight=Xweights)

print 'Loading Best Model'
h = load_model('best_model_cnn.hdf5')

#use new model to predict
print 'Predicting'

y_hat = model.predict(X[tr:])
y_hat = y_hat.flatten()
y_test = y[tr:]

#visualizing
print 'Visualizing'

from matplotlib import pyplot as plt
colormap = 'PuBuGn' #YlOrRd

def makeColorPlot(image, title, directory):
    plt.clf()
    plt.imshow(image, interpolation="none", cmap=colormap)
    plt.title(title, fontsize=15)
    plt.colorbar()
    plt.savefig(directory)

print 'Making Samples'

x_image = X[tr:tr+1, 0][0]
makeColorPlot(x_image, 'Sample Image', 'vis/samples/sample_image.png')

layer1_output = np.asarray(layer_1_vis([0]+[X[tr:tr+1]]))
makeColorPlot(layer1_output[0, 0, 0], 'Sample First Layer Activation Layer', 'vis/samples/sample_first_layer.png')

layer2_output = np.asarray(layer_2_vis([0]+[X[tr:tr+1]]))
makeColorPlot(layer2_output[0, 0, 0], 'Sample Second Layer Activation Layer', 'vis/samples/sample_second_layer.png')

filters = model.layers[0].kernel.get_value()
makeColorPlot(filters[:, :, 0, 0], 'First Layer Filter', 'vis/samples/sample_first_filters.png')

print 'Making Average'
average = X[tr:, 0]

average = np.sum(average, axis=0)
makeColorPlot(average, 'Average Image', 'vis/average_image.png')

print 'Making Layers'
def makeLayers(sample, subfolder):
    layer1_output = np.asarray(layer_1_vis([0]+[sample]))
    layer1_output = np.sum(layer1_output[0], axis=0)
    for i in range(layer1_output.shape[0]):
        makeColorPlot(layer1_output[i, 0:25, 0:25], 'First Layer Activation Layer (' + subfolder + ') ' + str(i), 'vis/activation_layers/layer1/' + subfolder + '/' + str(i) + '.png')

    layer2_output = np.asarray(layer_2_vis([0]+[sample]))
    layer2_output = np.sum(layer2_output[0], axis=0)
    for i in range(layer2_output.shape[0]):
        makeColorPlot(layer2_output[i, 0:25, 0:25], 'Second Layer Activation Layer (' + subfolder + ') ' + str(i), 'vis/activation_layers/layer2/' + subfolder + '/' + str(i) + '.png')

    layer3_output = np.asarray(layer_3_vis([0]+[sample]))
    layer3_output = np.sum(layer3_output, axis=1)
    layer3_output = np.reshape(layer3_output, (32, 32))
    makeColorPlot(layer3_output, 'Third Layer Activation Layer, Reshaped (' + subfolder + ') ', 'vis/activation_layers/layer3/' + subfolder + '.png')

#reconstruct bg and sig
sig = np.zeros((cvrts, 1, 32, 32))
bg = np.zeros((cvrt, 1, 32, 32))
bgi = 0
sgi = 0
for i in range(X.shape[0]):
    if y[i] == 0:
        bg[bgi, :, :, :] = X[i, :, :, :]
        bgi = bgi + 1
    else:
        sig[sgi, :, :, :] = X[i, :, :, :]
        sgi = sgi + 1

#memory limits, but we just need enough to get a good idea
sample_size = 10000

makeLayers(X[:sample_size], 'all')
makeLayers(sig[:sample_size], 'singlet')
makeLayers(bg[:sample_size], 'octet')

print 'Making Layer Differences'
sig = sig[:sample_size]
bg = bg[:sample_size]

layer1_output_s = np.asarray(layer_1_vis([0]+[sig]))
layer1_output_s = np.sum(layer1_output_s[0], axis=0)
layer1_output_b = np.asarray(layer_1_vis([0]+[bg]))
layer1_output_b = np.sum(layer1_output_b[0], axis=0)
layer1_output = layer1_output_s - layer1_output_b
for i in range(layer1_output.shape[0]):
    makeColorPlot(layer1_output[i, 0:32, 0:32], 'First Layer Activation Layer Differences ' + str(i), 'vis/activation_layers/layer1/dif/' + str(i) + '.png')

layer2_output_s = np.asarray(layer_2_vis([0]+[sig]))
layer2_output_s = np.sum(layer2_output_s[0], axis=0)
layer2_output_b = np.asarray(layer_2_vis([0]+[bg]))
layer2_output_b = np.sum(layer2_output_b[0], axis=0)
layer2_output = layer2_output_s - layer2_output_b
for i in range(layer2_output.shape[0]):
    makeColorPlot(layer2_output[i, 0:32, 0:32], 'Second Layer Activation Layer Differences ' + str(i), 'vis/activation_layers/layer2/dif/' + str(i) + '.png')

layer3_output_s = np.asarray(layer_3_vis([0]+[sig]))
layer3_output_s = np.sum(layer3_output_s[0], axis=0)
layer3_output_b = np.asarray(layer_3_vis([0]+[bg]))
layer3_output_b = np.sum(layer3_output_b[0], axis=0)
layer3_output = layer3_output_s - layer3_output_b
layer3_output = np.reshape(layer3_output, (32, 32))
makeColorPlot(layer3_output, 'Third Layer Activation Layer Differences, Reshaped', 'vis/activation_layers/layer3/dif.png')



print 'Making First Filters'
filters = model.layers[0].kernel.get_value()
for i in range(filters.shape[3]):
    makeColorPlot(filters[:, :, 0, i], 'First Layer Filters', 'vis/filters/'+ str(i) + '.png')

print 'Done!'