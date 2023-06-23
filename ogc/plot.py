import matplotlib.pyplot as plt
import numpy as np
import seaborn


def custom_hist(attr_index, xlabel, D, L, classesNames):
    # Function used to plot histograms. It receives the index of the attribute to plot,
    # the label for the x axis, the dataset matrix D, the array L with the values
    # for the classes and the list of classes names (used for the legend)
    plt.hist(D[attr_index, L == 0], color="#357ded",
             ec="#0000ff", density=True, alpha=0.6)
    plt.hist(D[attr_index, L == 1], color="#c00770",
             ec="#d2691e", density=True, alpha=0.6)
    plt.legend(classesNames)
    plt.xlabel(xlabel)
    plt.show()
    return


def custom_scatter(i, j, xlabel, ylabel, D, L, classesNames):
    # Function used for scatter plots. It receives the indexes i, j of the attributes
    # to plot, the labels for x, y axes, the dataset matrix D, the array L with the
    # values for the classes and the list of classes names (used for the legend)
    plt.scatter(D[i, L == 0], D[j, L == 0], color="#357ded", s=10)
    plt.scatter(D[i, L == 1], D[j, L == 1], color="#c00770", s=10)
    plt.legend(classesNames)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()
    return


def plotFeatures(D, L, featuresNames, classesNames):
    fig = plt.figure(figsize=(20, 20))
    fig.subplots_adjust(hspace=0.7, wspace=0.7)
    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            ax = fig.add_subplot(
                D.shape[0], D.shape[0], i * D.shape[0] + j + 1)
            if (i == j):
                # Then plot histogram
                ax.hist(D[i, L == 0], color="#357ded",
                        ec="#0000ff", density=True, alpha=0.4)
                ax.hist(D[i, L == 1], color="#c00770",
                        ec="#d2691e", density=True, alpha=0.4)
            else:
                # Else use scatter plot
                plt.scatter(D[i, L == 1], D[j, L == 1],
                            color="#c00770", s=1, alpha=0.4)
                plt.scatter(D[i, L == 0], D[j, L == 0],
                            color="#357ded", s=1, alpha=0.4)
    fig.legend(classesNames, )
    plt.show()
    return


def plotDCF(x, y, xlabel):
    plt.figure()
    plt.plot(x, y[0:len(x)], label='min DCF prior=0.5', color='b')
    plt.plot(x, y[len(x): 2*len(x)], label='min DCF prior=0.9', color='r')
    plt.plot(x, y[2*len(x): 3*len(x)], label='min DCF prior=0.1', color='g')
    plt.xlim([min(x), max(x)])
    plt.xscale("log")
    plt.legend(["min DCF prior=0.5", "min DCF prior=0.9", "min DCF prior=0.1"])
    plt.xlabel(xlabel)
    plt.ylabel("min DCF")
    return


def plotDCFpoly(x, y, xlabel):
    plt.figure()
    plt.plot(x, y[0:len(x)], label='min DCF prior=0.5 - c=0', color='b')
    plt.plot(x, y[len(x): 2*len(x)],
             label='min DCF prior=0.5 - c=1', color='r')
    plt.plot(x, y[2*len(x): 3*len(x)],
             label='min DCF prior=0.5 - c=10', color='g')
    plt.plot(x, y[3*len(x): 4*len(x)],
             label='min DCF prior=0.5 - c=30', color='m')

    plt.xlim([1e-5, 1e-1])
    plt.xscale("log")
    plt.legend(["min DCF prior=0.5 - c=0", "min DCF prior=0.5 - c=1",
                'min DCF prior=0.5 - c=10', 'min DCF prior=0.5 - c=30'])

    plt.xlabel(xlabel)
    plt.ylabel("min DCF")
    return


def plotDCFRBF(x, y, xlabel):
    plt.figure()
    plt.plot(x, y[0:len(x)], label='min DCF prior=0.5 - logγ=-5', color='b')
    plt.plot(x, y[len(x): 2*len(x)],
             label='min DCF prior=0.5 - logγ=-4', color='r')
    plt.plot(x, y[2*len(x): 3*len(x)],
             label='min DCF prior=0.5 - logγ=-3', color='g')

    plt.xlim([min(x), max(x)])
    plt.xscale("log")
    plt.legend(["min DCF prior=0.5 - logγ=-5", "min DCF prior=0.5 - logγ=-4",
                'min DCF prior=0.5 - logγ=-3'])

    plt.xlabel(xlabel)
    plt.ylabel("min DCF")
    return


def plotDCFGMM(x, y, xlabel):
    plt.figure()
    plt.plot(x, y[0:len(x)], label='min DCF prior=0.5', color='b')
    plt.plot(x, y[len(x): 2*len(x)], label='min DCF prior=0.9', color='r')
    plt.plot(x, y[2*len(x): 3*len(x)], label='min DCF prior=0.1', color='g')
    plt.xlim([min(x), max(x)])
    plt.xscale("log", basex=2)
    plt.legend(["min DCF prior=0.5", "min DCF prior=0.9", "min DCF prior=0.1"])

    plt.xlabel(xlabel)
    plt.ylabel("min DCF")
    return


def heatmap(D, L):
    fig = plt.figure(figsize=(15, 15))
    fig.subplots_adjust(wspace=0.7)
    ax = fig.add_subplot(1, 3, 1)
    ax.set_title("All classes")
    seaborn.heatmap(np.corrcoef(D), ax=ax, linewidth=0.2,
                    cmap="Greys", square=True, cbar=False)
    ax = fig.add_subplot(1, 3, 2)
    ax.set_title("Male class")
    seaborn.heatmap(np.corrcoef(
        D[:, L == 0]), ax=ax, linewidth=0.2, cmap="Blues", square=True, cbar=False)
    ax = fig.add_subplot(1, 3, 3)
    ax.set_title("Female class")
    seaborn.heatmap(np.corrcoef(
        D[:, L == 1]), ax=ax, linewidth=0.2, cmap="Reds", square=True, cbar=False)
    return


def bayesErrorPlot(dcf, mindcf, effPriorLogOdds, model):
    plt.figure()
    plt.plot(effPriorLogOdds, dcf, label='act DCF', color='r')
    plt.plot(effPriorLogOdds, mindcf, label='min DCF',
             color='b', linestyle="--")
    plt.xlim([min(effPriorLogOdds), max(effPriorLogOdds)])
    plt.legend([model + " - act DCF", model+" - min DCF"])
    plt.xlabel("prior log-odds")
    plt.ylabel("DCF")
    return


def bayesErrorPlotV2(dcf0, dcf1, mindcf, effPriorLogOdds, model, lambda0, lambda1):
    plt.figure()
    plt.plot(effPriorLogOdds, dcf0, label='act DCF', color='r')
    plt.plot(effPriorLogOdds, dcf1, label='act DCF', color='g')
    plt.plot(effPriorLogOdds, mindcf, label='min DCF',
             color='b', linestyle="--")
    plt.xlim([min(effPriorLogOdds), max(effPriorLogOdds)])
    plt.legend([model + " - act DCF lambda = "+lambda0, model +
               " - act DCF lambda = "+lambda1, model+" - min DCF"])
    plt.xlabel("prior log-odds")
    plt.ylabel("DCF")
    return


def plotROC(FPR, TPR, FPR1, TPR1, FPR2, TPR2):
    # Function used to plot TPR(FPR)
    plt.figure()
    plt.grid(linestyle='--')
    plt.plot(FPR, TPR, linewidth=2, color='r')
    plt.plot(FPR1, TPR1, linewidth=2, color='b')
    plt.plot(FPR2, TPR2, linewidth=2, color='g')
    plt.legend(["Tied-Cov", "Logistic regression",
               "GMM Full-Cov 8 components"])
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    return
