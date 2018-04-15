import numpy as np
import matplotlib
import matplotlib.pyplot as plt


ROC = 0 #make ROC plot with single threshold discrimnator if ROC = 1
flag = 0 #0 is all, 1 is use only deltar != 0
restrict_range = 1 #restrict to interesting range (1), or use entire range (0)
rbin = 3 #0 is no binning, 1 is <0.5, 2 is 0.5-0.7, 3 is >0.7 , all based on deltar
alter_binning = rbin + flag - rbin * flag #always use
weighting = 1 #0 is no weighting, 1 is weight mass
variable = 6 #jet pull is 0 and 1, jet mass is 6, delta_r is -1

if variable == 0:
    title = 'Jet Pull 1'
elif variable == 1:
    title  = 'Jet Pull 2'
elif variable == 6:
    title = 'Mass'
elif variable == -1:
    title = 'deltaR'
else:
    title = 'this is wrong and will error before the title is even needed'

if rbin == 1:
    flag = 0

if flag == 1:
    title = title + ', deltaR = 0 removed'

if weighting:
    title = title + ', mass reweighted'

#can change these to change savefile names
save = title + '_hist.png'
saveROC = title + '_ROC.png'

if rbin == 1:
    save = title + ', deltaR_low_bin' + '_hist.png'
    saveROC = title + ', deltaR_low_bin' + '_ROC.png'
    title = title + ', for 0 < deltaR < 0.5'
if rbin == 2:
    save = title + ', deltaR_peak_bin' + '_hist.png'
    saveROC = title + ', deltaR_peak_bin' + '_ROC.png'
    title = title + ', for 0.5 <= deltaR <= 0.7'
if rbin == 3:
    save = title + ', deltaR_high_bin' + '_hist.png'
    saveROC = title + ', deltaR_high_bin' + '_ROC.png'
    title = title + ', for deltaR > 0.7'


#load data
print 'Loading data...'
bgo = np.load('../Octet_Rotated_withDR_new.npy')
sigo = np.load('../Singlet_Rotated_withDR_new.npy')
bshape = bgo[:, variable].shape[0]
sshape = sigo[:, variable].shape[0]

#remove nans
mask = ~np.isnan(bgo).any(axis=1)
bgo = bgo[mask,...]
mask = ~np.isnan(sigo).any(axis=1)
sigo = sigo[mask,...]
    
#if flagged, use only delta_r!=0
if flag == 1:
    bgo = bgo[np.where(bgo[:, -1] != 0)]
    sigo = sigo[np.where(sigo[:, -1] != 0)]

#bin deltar
if rbin == 1:
    bgo = bgo[np.where((np.absolute(bgo[:, -1] - .25) < .25))]
    sigo = sigo[np.where((np.absolute(sigo[:, -1] - .25) < .25))]
if rbin == 2:
    bgo = bgo[np.where((np.absolute(bgo[:, -1] - .6) <= .1))]
    sigo = sigo[np.where((np.absolute(sigo[:, -1] - .6) <= .1))]    
if rbin == 3:
    bgo = bgo[np.where(bgo[:, -1] - .7 > 0)]
    sigo = sigo[np.where(sigo[:, -1] - .7 > 0)]

#make bins
if variable == 6:
    xmin = 0
    if restrict_range == 1:
        xmax = 300
    else:
        xmax = 400
    ymin = 1
    ymax = 100000
elif variable == 0 or variable == 1:
    xmin = -4
    xmax = 4
    ymin = 1
    ymax = 10000
elif variable == -1:
    xmin = 0
    xmax = 1.4
    ymin = 1
    ymax = 10000
bins = np.linspace(xmin, xmax, 100)

#mass bins
mxmin = 0
if restrict_range == 1:
    mxmax = 300
else:
    mxmax = 400
mymin = 1
mymax = 100000
mbins = np.linspace(mxmin, mxmax, 100)

#weighting
sweights = None
if weighting == 1:
    sweights = np.zeros(sigo.shape[0])

    for i in range(1, 100):
        locb = (bgo[:, 6] < mbins[i]) & (bgo[:, 6] >= mbins[i-1])
        locs = (sigo[:, 6] < mbins[i]) & (sigo[:, 6] >= mbins[i-1])
        n = bgo[locb].shape[0]
        d = sigo[locs].shape[0]
        if n==0 or d==0:
            sweights[locs] = 0
        else:
            sweights[locs] = float(n)/float(d)

#reshape
bg = bgo[:, variable]
sig = sigo[:, variable]

#make hist
hist_sig, sbins = np.histogram(sig, bins=bins, normed=False, weights=sweights)
hist_bg, bbins = np.histogram(bg, bins=bins, normed=False)

#plot histogram
matplotlib.rc('font', **{'size'   : 14})
plt.plot(bins[:-1], hist_sig, drawstyle='steps-post', color='blue', label='Singlet')
plt.plot(bins[:-1], hist_bg, drawstyle='steps-post', color='red', label='Octet')
plt.yscale('log')
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
plt.title(title, fontsize=20)
plt.legend(loc='upper right')
plt.savefig(save)
plt.clf()

#make ROCs
if ROC == 1:
    tpr = []
    fpr = []

    sigFreqSum = np.sum(hist_sig)
    bgFreqSum = np.sum(hist_bg)
    for j in range(hist_sig.shape[0]):
        if variable == 6:
            TP = np.sum(hist_sig[j:])
            FN = np.sum(hist_bg[j:])
            tpr.append(TP / float(sigFreqSum))
            fpr.append(FN / float(bgFreqSum))
        else:
            TP = np.sum(hist_sig[0:j+1])
            FN = np.sum(hist_bg[0:j+1])
            tpr.append(TP / float(sigFreqSum))
            fpr.append(FN / float(bgFreqSum))

    if variable != 6:
        tpr = [0.0] + tpr
        fpr = [0.0] + fpr

    tpr, fpr = np.asarray(tpr), np.asarray(fpr)

    #AUC
    ufpr = np.unique(fpr)
    utpr = []
    for k in range(len(ufpr)):
        ind = np.where(fpr == ufpr[k])
        utpr.append(np.max(tpr[ind]))

    utpr = np.asarray(utpr)

    AUC = np.trapz(utpr, x=ufpr)
    st_AUC = "AUC = " + str('%.2f' % round(AUC, 2))

    plt.plot(fpr, tpr, linewidth=2, color='blue', lw=2)   #drawstyle='steps-post'
    plt.plot([0,1],[0,1], 'r--', lw=1.5)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.text(0.55,0.07, st_AUC, fontsize=23, weight=550)
    plt.xlabel('false positive rate (fpr)', fontsize=17)
    plt.ylabel('true positive rate (tpr)', fontsize=17)
    plt.title(title, fontsize=20)

    plt.savefig(saveROC)

print 'Done!'