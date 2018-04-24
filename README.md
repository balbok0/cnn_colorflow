# cnn_colorflow

Some of these files should be split up and reorginized.

Pearson.py makes pearson correlation coefficient images (PCC) for the neural network, comparing its output and the truth value.
PearsonVar.py does this with various meta-variables.

cnn.py has the basic cnn.
cnn_visualizer.py does the visualization work. Basically just making images of the neural network's layers.
lCurve.py makes learning curves.

plot_ROC.py is an intial attempt to seperate things out. It plots the ROC using the tpr and fpr.
