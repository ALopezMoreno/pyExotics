import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from cycler import cycler
import scipy.optimize as so
import time
from scipy.ndimage.filters import gaussian_filter
import matplotlib.lines as mlines
import matplotlib as mpl
from matplotlib import colors
import matplotlib.patches as patches

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]})


from matplotlib.colors import LinearSegmentedColormap

cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905],
 [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143],
 [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952,
  0.779247619], [0.1252714286, 0.3242428571, 0.8302714286],
 [0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238,
  0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571],
 [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571,
  0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429],
 [0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667,
  0.8467], [0.0779428571, 0.5039857143, 0.8383714286],
 [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571,
  0.8262714286], [0.0640571429, 0.5569857143, 0.8239571429],
 [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524,
  0.819852381], [0.0265, 0.6137, 0.8135], [0.0238904762, 0.6286619048,
  0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667],
 [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381,
  0.7607190476], [0.0383714286, 0.6742714286, 0.743552381],
 [0.0589714286, 0.6837571429, 0.7253857143],
 [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429],
 [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429,
  0.6424333333], [0.2178285714, 0.7250428571, 0.6192619048],
 [0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619,
  0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667],
 [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524,
  0.5033142857], [0.4871238095, 0.7490619048, 0.4839761905],
 [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476,
  0.4493904762], [0.609852381, 0.7473142857, 0.4336857143],
 [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333],
 [0.7184095238, 0.7411333333, 0.3904761905],
 [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667,
  0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762],
 [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217],
 [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857,
  0.2886428571], [0.9738952381, 0.7313952381, 0.266647619],
 [0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857,
  0.2164142857], [0.9955333333, 0.7860571429, 0.196652381],
 [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857],
 [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309],
 [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333,
  0.0948380952], [0.9661, 0.9514428571, 0.0755333333],
 [0.9763, 0.9831, 0.0538]]

parula_map = LinearSegmentedColormap.from_list('parula', cm_data)
parula_map_r = LinearSegmentedColormap.from_list('parula_r', np.flip(cm_data, axis=0))

# Make a simple but good-looking line plot
def niceLinPlot(ax, xdata, ydata, logx=True, logy=True, **kwargs):
    ax.plot(xdata, ydata, **kwargs)
    if logy:
         ax.set_yscale('log')
    if logx:
         ax.set_xscale('log')
    return 0

# Make a simple but good-looking contour map
def plot2Dcontour(ax, X, Y, data, logx=False, logy=False, cont=False, percentile=None, reverse=False, **kwargs):
    if reverse:
        mycmap=parula_map_r
    else:
        mycmap=parula_map

    a = ax.contourf(X, Y, data, 100, cmap=mycmap, **kwargs) #YlGnBu_r
    if logy:
         ax.set_yscale('log')
    if logx:
         ax.set_xscale('log')
    if percentile:
        if not reverse:
            percentile= 100-np.asarray(percentile)
        line = np.percentile(data, percentile)
        ax.contour(X,Y,data,levels=line, colors=('r',),linestyles=('-',),linewidths=(1,))
    return a



# Plot histograms of |U_ij|
def plotPMNS(mixing_matrices, abs=True, Jpreset=True, lowlim=-0.1, highlim=0.1, **kwargs):

    if abs:
        # Mixing matrixces is an array of complex 3x3 matrices
        ModMatrix = np.abs(mixing_matrices)
    else:
        ModMatrix = mixing_matrices

    fig, axs = plt.subplots(nrows=3, ncols=3, dpi=500, sharex=False, sharey=False)

    if Jpreset:
        n, bins, patches = axs[0, 0].hist(ModMatrix[:, 0, 0], facecolor='#2ab0ff',  alpha=0.7, **kwargs)
        n = n.astype('int')  # it MUST be integer# Good old loop. Choose colormap of your taste
        for i in range(len(patches)):
            patches[i].set_facecolor(parula_map(n[i] / max(n)))
        n, bins, patches = axs[0, 1].hist(ModMatrix[:, 0, 1],  facecolor='#2ab0ff',  alpha=0.7, **kwargs)
        n = n.astype('int')  # it MUST be integer# Good old loop. Choose colormap of your taste
        for i in range(len(patches)):
            patches[i].set_facecolor(parula_map(n[i] / max(n)))
        n, bins, patches = axs[0, 2].hist(ModMatrix[:, 0, 2],  facecolor='#2ab0ff',  alpha=0.7, **kwargs)
        n = n.astype('int')  # it MUST be integer# Good old loop. Choose colormap of your taste
        for i in range(len(patches)):
            patches[i].set_facecolor(parula_map(n[i] / max(n)))

        n, bins, patches = axs[1, 0].hist(ModMatrix[:, 1, 0],  facecolor='#2ab0ff',  alpha=0.7, **kwargs)
        n = n.astype('int')  # it MUST be integer# Good old loop. Choose colormap of your taste
        for i in range(len(patches)):
            patches[i].set_facecolor(parula_map(n[i] / max(n)))
        n, bins, patches = axs[1, 1].hist(ModMatrix[:, 1, 1],  facecolor='#2ab0ff',  alpha=0.7, **kwargs)
        n = n.astype('int')  # it MUST be integer# Good old loop. Choose colormap of your taste
        for i in range(len(patches)):
            patches[i].set_facecolor(parula_map(n[i] / max(n)))
        n, bins, patches = axs[1, 2].hist(ModMatrix[:, 1, 2],  facecolor='#2ab0ff',  alpha=0.7, **kwargs)
        n = n.astype('int')  # it MUST be integer# Good old loop. Choose colormap of your taste
        for i in range(len(patches)):
            patches[i].set_facecolor(parula_map(n[i] / max(n)))

        n, bins, patches = axs[2, 0].hist(ModMatrix[:, 2, 0],  facecolor='#2ab0ff',  alpha=0.7, **kwargs)
        n = n.astype('int')  # it MUST be integer# Good old loop. Choose colormap of your taste
        for i in range(len(patches)):
            patches[i].set_facecolor(parula_map(n[i] / max(n)))
        n, bins, patches = axs[2, 1].hist(ModMatrix[:, 2, 1],  facecolor='#2ab0ff',  alpha=0.7, **kwargs)
        n = n.astype('int')  # it MUST be integer# Good old loop. Choose colormap of your taste
        for i in range(len(patches)):
            patches[i].set_facecolor(parula_map(n[i] / max(n)))
        n, bins, patches = axs[2, 2].hist(ModMatrix[:, 2, 2],  facecolor='#2ab0ff',  alpha=0.7, **kwargs)
        n = n.astype('int')  # it MUST be integer# Good old loop. Choose colormap of your taste
        for i in range(len(patches)):
            patches[i].set_facecolor(parula_map(n[i] / max(n)))

        for i in range(3):
            for j in range(3):
                axs[i, j].set_xlim(lowlim, highlim)
                axs[i, j].set_yticks([])
                axs[i, j].set_xticks([])
                axs[2, j].xaxis.set_major_locator(ticker.LinearLocator(5))
                axs[2, j].set_xticks(np.linspace(lowlim, highlim, 5)[1:-1])
                axs[2, j].xaxis.set_minor_locator(ticker.LinearLocator(25))
                axs[2, j].set_xticklabels(np.linspace(lowlim, highlim, 5)[1:-1], fontsize=7)
                axs[2, j].xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
                axs[2, j].tick_params(axis='x')
                axs[2, j].tick_params(which='both', direction="inout")
    else:
        axs[0, 0].hist(ModMatrix[:, 0, 0], **kwargs)
        axs[0, 1].hist(ModMatrix[:, 0, 1], **kwargs)
        axs[0, 2].hist(ModMatrix[:, 0, 2], **kwargs)

        axs[1, 0].hist(ModMatrix[: 1, 0], **kwargs)
        axs[1, 1].hist(ModMatrix[: 1, 1], **kwargs)
        axs[1, 2].hist(ModMatrix[: 1, 2], **kwargs)

        axs[2, 0].hist(ModMatrix[: 2, 0], **kwargs)
        axs[2, 1].hist(ModMatrix[: 2, 1], **kwargs)
        axs[2, 2].hist(ModMatrix[: 2, 2], **kwargs)

    axs[0, 0].set_title(r"$m_1$", fontsize=12)
    axs[0, 1].set_title(r"$m_2$", fontsize=12)
    axs[0, 2].set_title(r"$m_3$", fontsize=12)

    axs[0, 0].set_ylabel(r"$e$", fontsize=12)
    axs[1, 0].set_ylabel(r"$\mu$", fontsize=12)
    axs[2, 0].set_ylabel(r"$\tau$", fontsize=12)
    for i in range(3):
        for j in range(3):
            axs[i, j].set_box_aspect(1)
    plt.tight_layout(pad=2, w_pad=-11.5, h_pad=-0.20)
# FUNCTIONS FOR PLOTTING MU-TAU UNITARITY TRIANGLES

def PlotTriPost(s12, s23, s13, dcp, w, bnumber, r=0, s=0, third=False, colormap='viridis', normalisation=mpl.colors.Normalize(),
            title='', asimovMode='', asimovValues=[], best_fit=[], contours=False):
    fig, axs = plt.subplots(nrows=1, ncols=3, dpi=400, figsize=(8, 3.5))

    cmap = cm.get_cmap(colormap, 256)
    newcolors = cmap(np.linspace(0, 1, 256))
    white = np.array([ 1, 1, 1, 1])
    newcolors[:1, :] = white
    colormap = ListedColormap(newcolors)

    if title:
        if asimovMode:
            fig.suptitle(title + " \n \n Asimov: " + asimovMode,
                         multialignment='left', fontsize=15)
        else:
            fig.suptitle(title, fontsize=15)

    # U_pmns_R
    U_e1_R, U_e2_R, U_e3_R, U_m1_R, U_m2_R, U_m3_R, U_t1_R, U_t2_R, U_t3_R = pars.load_real_nominal_PMNS(s12, s23, s13,
                                                                                                         dcp, r, s)

    # U_pmns_I
    U_e1_I, U_e2_I, U_e3_I, U_m1_I, U_m2_I, U_m3_I, U_t1_I, U_t2_I, U_t3_I = pars.load_imag_nominal_PMNS(s12, s23, s13,
                                                                                                         dcp, r, s)
    # Onto the actual vertices
    TRI1_V1 = np.asarray([np.multiply(U_e2_R, U_e3_R) - np.multiply(U_e2_I, -U_e3_I),
                          np.multiply(U_e2_R, -U_e3_I) + np.multiply(U_e2_I, U_e3_R)])

    TRI1_V2 = np.add(TRI1_V1, np.asarray([np.multiply(U_m2_R, U_m3_R) - np.multiply(U_m2_I, -U_m3_I),
                                          np.multiply(U_m2_R, -U_m3_I) + np.multiply(U_m2_I, U_m3_R)]))

    TRI1_V3 = np.add(TRI1_V2, np.asarray([np.multiply(U_t2_R, U_t3_R) - np.multiply(U_t2_I, -U_t3_I),
                                          np.multiply(U_t2_R, -U_t3_I) + np.multiply(U_t2_I, U_t3_R)]))


    TRI2_V1 = np.asarray([np.multiply(U_e3_R, U_e1_R) - np.multiply(U_e3_I, -U_e1_I),
                          np.multiply(U_e3_R, -U_e1_I) + np.multiply(U_e3_I, U_e1_R)])

    TRI2_V2 = np.add(TRI2_V1, np.asarray([np.multiply(U_m3_R, U_m1_R) - np.multiply(U_m3_I, -U_m1_I),
                                          np.multiply(U_m3_R, -U_m1_I) + np.multiply(U_m3_I, U_m1_R)]))


    TRI2_V3 = np.add(TRI2_V2, np.asarray([np.multiply(U_t3_R, U_t1_R) - np.multiply(U_t3_I, -U_t1_I),
                                          np.multiply(U_t3_R, -U_t1_I) + np.multiply(U_t3_I, U_t1_R)]))


    TRI3_V1 = np.asarray([np.multiply(U_e1_R, U_e2_R) - np.multiply(U_e1_I, -U_e2_I),
                          np.multiply(U_e1_R, -U_e2_I) + np.multiply(U_e1_I, U_e2_R)])


    TRI3_V2 = np.add(TRI3_V1, np.asarray([np.multiply(U_m1_R, U_m2_R) - np.multiply(U_m1_I, -U_m2_I),
                                          np.multiply(U_m1_R, -U_m2_I) + np.multiply(U_m1_I, U_m2_R)]))


    TRI3_V3 = np.add(TRI3_V2, np.asarray([np.multiply(U_t1_R, U_t2_R) - np.multiply(U_t1_I, -U_t2_I),
                                          np.multiply(U_t1_R, -U_t2_I) + np.multiply(U_t1_I, U_t2_R)]))


    if third != True:
        ws = np.concatenate((w, w))
        T1 = np.concatenate((TRI1_V1, TRI1_V2), axis=1)
        T2 = np.concatenate((TRI2_V1, TRI2_V2), axis=1)
        T3 = np.concatenate((TRI3_V1, TRI3_V2), axis=1)

    else:
        ws = np.concatenate((w, w, w))
        T1 = np.concatenate((TRI1_V1, TRI1_V2, TRI1_V3), axis=1)
        T2 = np.concatenate((TRI2_V1, TRI2_V2, TRI2_V3), axis=1)
        T3 = np.concatenate((TRI3_V1, TRI3_V2, TRI3_V3), axis=1)


    axs[0].hist2d(T1[0, :], T1[1, :], bins=bnumber, weights=ws, cmap=colormap, norm=normalisation)
                  #range=np.array([(-0.5, 0.5), (-0.5, 0.5)]))
    axs[1].hist2d(T2[0, :], T2[1, :], bins=bnumber, weights=ws, cmap=colormap, norm=normalisation)
                  #range=np.array([(-0.5, 0.5), (-0.5, 0.5)]))
    axs[2].hist2d(T3[0, :], T3[1, :], bins=bnumber, weights=ws, cmap=colormap, norm=normalisation)
                  #range=np.array([(-0.5, 0.5), (-0.5, 0.5)]))




    absmin = []
    absmax = []
    for i in range(3):
        ymin, ymax = axs[i].get_ylim()
        xmin, xmax = axs[i].get_xlim()
        absmin.append(np.min([xmin, ymin]))
        absmax.append(np.max([xmax, ymax]))
        axs[i].axvline(x=0, color='black', linestyle='dashed', linewidth=0.8, alpha=0.5)
        axs[i].axhline(y=0, color='black', linestyle='dashed', linewidth=0.8, alpha=0.5)


    if contours == True:

        T1V1 = contour(TRI1_V1[0, :], TRI1_V1[1, :], int(bnumber), 0.90, 0.68, np.array([(absmin[0], absmax[0]), (absmin[0], absmax[0])]), weights=w)
        T1V2 = contour(TRI1_V2[0, :], TRI1_V2[1, :], int(bnumber), 0.90, 0.68, np.array([(absmin[0], absmax[0]), (absmin[0], absmax[0])]), weights=w)
        T1V3 = contour(TRI1_V3[0, :], TRI1_V3[1, :], int(bnumber), 0.90, 0.68, np.array([(absmin[0], absmax[0]), (absmin[0], absmax[0])]), weights=w)

        T2V1 = contour(TRI2_V1[0, :], TRI2_V1[1, :], int(bnumber), 0.90, 0.68, np.array([(absmin[1], absmax[1]), (absmin[1], absmax[1])]), weights=w)
        T2V2 = contour(TRI2_V2[0, :], TRI2_V2[1, :], int(bnumber), 0.90, 0.68, np.array([(absmin[1], absmax[1]), (absmin[1], absmax[1])]), weights=w)
        T2V3 = contour(TRI2_V3[0, :], TRI2_V3[1, :], int(bnumber), 0.90, 0.68, np.array([(absmin[1], absmax[1]), (absmin[1], absmax[1])]), weights=w)

        T3V1 = contour(TRI3_V1[0, :], TRI3_V1[1, :], int(bnumber), 0.90, 0.68, np.array([(absmin[2], absmax[2]), (absmin[2], absmax[2])]), weights=w)
        T3V2 = contour(TRI3_V2[0, :], TRI3_V2[1, :], int(bnumber), 0.90, 0.68, np.array([(absmin[2], absmax[2]), (absmin[2], absmax[2])]), weights=w)
        T3V3 = contour(TRI3_V3[0, :], TRI3_V3[1, :], int(bnumber), 0.90, 0.68, np.array([(absmin[2], absmax[2]), (absmin[2], absmax[2])]), weights=w)

        if third != True:
            CT1 = np.concatenate((T1V1, T1V2), axis=0)
            CT2 = np.concatenate((T2V1, T2V2), axis=0)
            CT3 = np.concatenate((T3V1, T3V2), axis=0)
        else:
            CT1 = np.concatenate((T1V1, T1V2, T1V3), axis=0)
            CT2 = np.concatenate((T2V1, T2V2, T2V3), axis=0)
            CT3 = np.concatenate((T3V1, T3V2, T3V3), axis=0)


        axs[0].hist2d(CT1[:, 0], CT1[:, 1], bins=bnumber, cmap=colormap, # norm=normalisation,
                      range=np.array([(absmin[0], absmax[0]), (absmin[0], absmax[0])]))
        axs[1].hist2d(CT2[:, 0], CT2[:, 1], bins=bnumber, cmap=colormap, # norm=normalisation,
                      range=np.array([(absmin[1], absmax[1]), (absmin[1], absmax[1])]))
        axs[2].hist2d(CT3[:, 0], CT3[:, 1], bins=bnumber, cmap=colormap, # norm=normalisation,
                      range=np.array([(absmin[2], absmax[2]), (absmin[2], absmax[2])]))



    else:
        axs[0].hist2d(T1[0, :], T1[1, :], bins=bnumber, weights=ws, cmap=colormap, norm=normalisation,
                      range=np.array([(-0.5, 0.5), (-0.5, 0.5)]))
        axs[1].hist2d(T2[0, :], T2[1, :], bins=bnumber, weights=ws, cmap=colormap, norm=normalisation,
                      range=np.array([(-0.5, 0.5), (-0.5, 0.5)]))
        axs[2].hist2d(T3[0, :], T3[1, :], bins=bnumber, weights=ws, cmap=colormap, norm=normalisation,
                      range=np.array([(-0.5, 0.5), (-0.5, 0.5)]))

    if len(asimovValues) != 0:

        s12 = asimovValues[0]
        s23 = asimovValues[1]
        s13 = asimovValues[2]
        dcp = asimovValues[3]
        # U_pmns_R
        U_e1_R, U_e2_R, U_e3_R, U_m1_R, U_m2_R, U_m3_R, U_t1_R, U_t2_R, U_t3_R = pars.load_real_nominal_PMNS(s12, s23,
                                                                                                             s13,
                                                                                                             dcp, r, s)

        # U_pmns_I
        U_e1_I, U_e2_I, U_e3_I, U_m1_I, U_m2_I, U_m3_I, U_t1_I, U_t2_I, U_t3_I = pars.load_imag_nominal_PMNS(s12, s23,
                                                                                                             s13,
                                                                                                             dcp, r, s)
        # Onto the actual vertices
        TRI1_V1 = np.asarray([np.multiply(U_e2_R, U_e3_R) - np.multiply(U_e2_I, -U_e3_I),
                              np.multiply(U_e2_R, -U_e3_I) + np.multiply(U_e2_I, U_e3_R)])
        TRI1_V2 = np.add(TRI1_V1, np.asarray([np.multiply(U_m2_R, U_m3_R) - np.multiply(U_m2_I, -U_m3_I),
                                              np.multiply(U_m2_R, -U_m3_I) + np.multiply(U_m2_I, U_m3_R)]))
        TRI1_V3 = np.add(TRI1_V2, np.asarray([np.multiply(U_t2_R, U_t3_R) - np.multiply(U_t2_I, -U_t3_I),
                                              np.multiply(U_t2_R, -U_t3_I) + np.multiply(U_t2_I, U_t3_R)]))

        TRI2_V1 = np.asarray([np.multiply(U_e3_R, U_e1_R) - np.multiply(U_e3_I, -U_e1_I),
                              np.multiply(U_e3_R, -U_e1_I) + np.multiply(U_e3_I, U_e1_R)])
        TRI2_V2 = np.add(TRI2_V1, np.asarray([np.multiply(U_m3_R, U_m1_R) - np.multiply(U_m3_I, -U_m1_I),
                                              np.multiply(U_m3_R, -U_m1_I) + np.multiply(U_m3_I, U_m1_R)]))
        TRI2_V3 = np.add(TRI2_V2, np.asarray([np.multiply(U_t3_R, U_t1_R) - np.multiply(U_t3_I, -U_t1_I),
                                              np.multiply(U_t3_R, -U_t1_I) + np.multiply(U_t3_I, U_t1_R)]))

        TRI3_V1 = np.asarray([np.multiply(U_e1_R, U_e2_R) - np.multiply(U_e1_I, -U_e2_I),
                              np.multiply(U_e1_R, -U_e2_I) + np.multiply(U_e1_I, U_e2_R)])
        TRI3_V2 = np.add(TRI3_V1, np.asarray([np.multiply(U_m1_R, U_m2_R) - np.multiply(U_m1_I, -U_m2_I),
                                              np.multiply(U_m1_R, -U_m2_I) + np.multiply(U_m1_I, U_m2_R)]))
        TRI3_V3 = np.add(TRI3_V2, np.asarray([np.multiply(U_t1_R, U_t2_R) - np.multiply(U_t1_I, -U_t2_I),
                                              np	.multiply(U_t1_R, -U_t2_I) + np.multiply(U_t1_I, U_t2_R)]))
        aT1 = np.array(([0, 0], TRI1_V1, TRI1_V2, TRI1_V3))
        aT2 = np.array(([0, 0], TRI2_V1, TRI2_V2, TRI2_V3))
        aT3 = np.array(([0, 0], TRI3_V1, TRI3_V2, TRI3_V3))
        axs[0].plot(aT1[:, 0], aT1[:, 1], 'r-', linewidth = 0.7)
        axs[1].plot(aT2[:, 0], aT2[:, 1], 'r-', linewidth = 0.7)
        axs[2].plot(aT3[:, 0], aT3[:, 1], 'r-', linewidth = 0.7, label='Asimov triangle')

    if len(best_fit) != 0:
        s12 = best_fit[0]
        s23 = best_fit[1]
        s13 = best_fit[2]
        dcp = best_fit[3]
        # U_pmns_R
        U_e1_R, U_e2_R, U_e3_R, U_m1_R, U_m2_R, U_m3_R, U_t1_R, U_t2_R, U_t3_R = pars.load_real_nominal_PMNS(s12, s23,
                                                                                                             s13,
                                                                                                             dcp, r, s)

        # U_pmns_I
        U_e1_I, U_e2_I, U_e3_I, U_m1_I, U_m2_I, U_m3_I, U_t1_I, U_t2_I, U_t3_I = pars.load_imag_nominal_PMNS(s12, s23,
                                                                                                             s13,
                                                                                                             dcp, r, s)
        # Onto the actual vertices
        TRI1_V1 = np.asarray([np.multiply(U_e2_R, U_e3_R) - np.multiply(U_e2_I, -U_e3_I),
                              np.multiply(U_e2_R, -U_e3_I) + np.multiply(U_e2_I, U_e3_R)])
        TRI1_V2 = np.add(TRI1_V1, np.asarray([np.multiply(U_m2_R, U_m3_R) - np.multiply(U_m2_I, -U_m3_I),
                                              np.multiply(U_m2_R, -U_m3_I) + np.multiply(U_m2_I, U_m3_R)]))
        TRI1_V3 = np.add(TRI1_V2, np.asarray([np.multiply(U_t2_R, U_t3_R) - np.multiply(U_t2_I, -U_t3_I),
                                              np.multiply(U_t2_R, -U_t3_I) + np.multiply(U_t2_I, U_t3_R)]))

        TRI2_V1 = np.asarray([np.multiply(U_e3_R, U_e1_R) - np.multiply(U_e3_I, -U_e1_I),
                              np.multiply(U_e3_R, -U_e1_I) + np.multiply(U_e3_I, U_e1_R)])
        TRI2_V2 = np.add(TRI2_V1, np.asarray([np.multiply(U_m3_R, U_m1_R) - np.multiply(U_m3_I, -U_m1_I),
                                              np.multiply(U_m3_R, -U_m1_I) + np.multiply(U_m3_I, U_m1_R)]))
        TRI2_V3 = np.add(TRI2_V2, np.asarray([np.multiply(U_t3_R, U_t1_R) - np.multiply(U_t3_I, -U_t1_I),
                                              np.multiply(U_t3_R, -U_t1_I) + np.multiply(U_t3_I, U_t1_R)]))

        TRI3_V1 = np.asarray([np.multiply(U_e1_R, U_e2_R) - np.multiply(U_e1_I, -U_e2_I),
                              np.multiply(U_e1_R, -U_e2_I) + np.multiply(U_e1_I, U_e2_R)])
        TRI3_V2 = np.add(TRI3_V1, np.asarray([np.multiply(U_m1_R, U_m2_R) - np.multiply(U_m1_I, -U_m2_I),
                                              np.multiply(U_m1_R, -U_m2_I) + np.multiply(U_m1_I, U_m2_R)]))
        TRI3_V3 = np.add(TRI3_V2, np.asarray([np.multiply(U_t1_R, U_t2_R) - np.multiply(U_t1_I, -U_t2_I),
                                              np.multiply(U_t1_R, -U_t2_I) + np.multiply(U_t1_I, U_t2_R)]))
        aT1 = np.array(([0, 0], TRI1_V1, TRI1_V2, TRI1_V3))
        aT2 = np.array(([0, 0], TRI2_V1, TRI2_V2, TRI2_V3))
        aT3 = np.array(([0, 0], TRI3_V1, TRI3_V2, TRI3_V3))
        axs[0].plot(aT1[:, 0], aT1[:, 1], 'b-', linewidth = 0.7, alpha=0.7)
        axs[1].plot(aT2[:, 0], aT2[:, 1], 'b-', linewidth = 0.7, alpha=0.7)
        axs[2].plot(aT3[:, 0], aT3[:, 1], 'b-', linewidth = 0.7, alpha=0.7, label='Best fit triangle')

    for i in range(3):
        axs[i].set_xlim([absmin[i], absmax[i]])
        axs[i].set_ylim([absmin[i], absmax[i]])
        axs[i].set_box_aspect(1)
        axs[i].xaxis.set_major_locator(ticker.LinearLocator(13))
        axs[i].set_xticks([axs[i].get_xticks()[0],
                           axs[i].get_xticks()[6],
                           axs[i].get_xticks()[12]])
        axs[i].xaxis.set_minor_locator(ticker.LinearLocator(25))
        axs[i].yaxis.set_major_locator(ticker.LinearLocator(13))
        axs[i].set_yticks([axs[i].get_yticks()[0],
                           axs[i].get_yticks()[6],
                           axs[i].get_yticks()[12]])
        axs[i].yaxis.set_minor_locator(ticker.LinearLocator(25))
        axs[i].xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        axs[i].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        axs[i].tick_params(which='both', direction="inout")

    title0 = r"$\Delta_1 : U_{e2}U^{*}_{e3} + U_{\mu 2}U^{*}_{\mu 3} + U_{\tau 2}U^{*}_{\tau 3}$"
    title1 = r"$\Delta_2 : U_{e3}U^{*}_{e1} + U_{\mu 3}U^{*}_{\mu 1} + U_{\tau 3}U^{*}_{\tau 1}$"
    title2 = r"$\Delta_3 : U_{e1}U^{*}_{e2} + U_{\mu 1}U^{*}_{\mu 2} + U_{\tau 1}U^{*}_{\tau 2}$"

    axs[0].set_title(title0, fontsize=9)
    axs[1].set_title(title1, fontsize=9)
    axs[2].set_title(title2, fontsize=9)
    axs[2].legend(bbox_to_anchor= [0.9,1.4])

    plt.tight_layout()

    plt.savefig("./images/Majorana_triangles_" + title + "--" + str(r) + '_' + str(s) +
                ".png")
    plt.show()
    return ()

# DRAW A LINE BETWEEN TWO VERTICAL POINTS
def draw_line_between_verticals(ax, x1, x2, draw_arrow=False, height=0.5, text=None, thickness=1, xscale='linear', prefactor=10**-2):
    ymin, ymax = ax.get_ylim()
    ypos = ymin + height * (ymax - ymin)
    if xscale == 'log':
        x1l, x2l = np.log10([x1, x2])
    else:
        x1l, x2l = [x1, x2]
    ax.plot([x1, x1], [ymin, ymax], 'k-', lw=1)
    ax.plot([x2, x2], [ymin, ymax], 'k-', lw=1)
    if draw_arrow:
        ax.arrow(x2, ypos, 0.99*(-np.abs(x2-x1)), 0, head_width=0.03*(ymax-ymin), head_length=-prefactor*(x1-x2), fc='k', ec='k', lw=thickness, length_includes_head=True)
        ax.plot([x1, x2], [ypos, ypos], 'k-', lw=thickness)

    else:
        ax.plot([x1, x2], [ypos, ypos], 'k-', lw=thickness)
    if text:
        text_height = height + 0.05 * (ymin - ymax)
        text_width = (x2 - x1) * 0.2
        if xscale == 'log':
            x1, x2 = np.log10([x1, x2])
            text_x = np.mean([x1, x2])
            text_y = text_height
        else:
            text_x = (x1 + x2) / 2
            text_y = text_height
        rect = patches.Rectangle(
            (text_x - text_width / 2, text_height), text_width, 0.05 * (ymin - ymax),
            linewidth=1,
            edgecolor='gray',
            facecolor='white',
            alpha=1,
            zorder=10,
            transform=ax.transData,
            clip_on=False
        )
        ax.add_patch(rect)
        ax.text(text_x, text_y, text, ha='center', va='bottom', transform=ax.transData, zorder=11, clip_on=False)


