
import numpy as np
import matplotlib.pyplot as plt

tpr = np.genfromtxt('CSVs/tpr.csv', delimiter=',')
fpr = np.genfromtxt('CSVs/fpr.csv', delimiter=',')

AUC = np.trapz(tpr, x=fpr)
st_AUCb = "AUC = " + str('%.3f' % round(AUC, 3))

pltRoc = plt.plot(fpr, tpr, linewidth=2, drawstyle='steps-post', color='blue')
pltDiag = plt.plot([0,1],[0,1], 'r--')

plt.text(0.6,0.2, st_AUCb, fontsize=17, weight=550)
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('false positive rate (fpr)', fontsize=15)
plt.ylabel('true positive rate (tpr)', fontsize=15)
plt.title('Gradient Boost on two of my meta variables', fontsize=19)
plt.savefig('ROC.png')

#plt.clf()

#tpr = np.genfromtxt('CSVs/tpr_o.csv', delimiter=',')
#fpr = np.genfromtxt('CSVs/fpr_o.csv', delimiter=',')

#AUC = np.trapz(tpr, x=fpr)
#st_AUCb = "AUC = " + str('%.3f' % round(AUC, 3))

#pltRoc = plt.plot(fpr, tpr, linewidth=2, drawstyle='steps-post', color='blue')
#pltDiag = plt.plot([0,1],[0,1], 'r--')

#plt.text(0.6,0.2, st_AUCb, fontsize=17, weight=550)
#plt.xlim([0,1])
#plt.ylim([0,1])
#plt.xlabel('false positive rate (fpr)', fontsize=15)
#plt.ylabel('true positive rate (tpr)', fontsize=15)
#plt.title('CNN Performance on Training Set', fontsize=19)
#plt.savefig('ROC_o.png')