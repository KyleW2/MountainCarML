import pickle
from turtle import color
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

def plotActionPolicy(filename, ax):
    f = open(filename, "rb")
    policy = pickle.load(f)

    x = []
    y = []
    z = []

    def f(x,y):
        t = (x, y)
        return policy[t].actionValues[policy[t].getMaxAction()]

    for state in policy.values():
        x.append(state.position)
        y.append(state.velocity)
        z.append(state.actionValues[state.getMaxAction()])

    #X, Y = np.meshgrid(x, y)
    #Z = f(x,y)
    ax.scatter(x, y, z, edgecolors='blue')

def plotStatePolicy(filename, ax):
    f = open(filename, "rb")
    policy = pickle.load(f)

    x = []
    y = []
    z = []

    def f(x,y):
        t = (x, y)
        return policy[t].actionValues[policy[t].getMaxAction()]

    for state in policy.values():
        x.append(state.position)
        y.append(state.velocity)
        z.append(state.getValue())

    #X, Y = np.meshgrid(x, y)
    #Z = f(x,y)
    ax.scatter(x, y, z, edgecolors='red')
    
def plot_curve(data_list, filepath="./my_plot.png",
            x_label="X", y_label="Y",
            x_range=(0, 1), y_range=(0,1), color="-r", kernel_size=50, alpha=0.4, grid=True):
    """Plot a graph using matplotlib

    """
    if(len(data_list) <=1):
        print("[WARNING] the data list is empty, no plot will be saved.")
        return
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=True)
    ax.grid(grid)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.plot(data_list, color, alpha=alpha)  # The original data is showed in background
    kernel = np.ones(int(kernel_size))/float(kernel_size)  # Smooth the graph using a convolution
    tot_data = len(data_list)
    lower_boundary = int(kernel_size/2.0)
    upper_boundary = int(tot_data-(kernel_size/2.0))
    data_convolved_array = np.convolve(data_list, kernel, 'same')[lower_boundary:upper_boundary]
    #print("arange: " + str(np.arange(tot_data)[lower_boundary:upper_boundary]))
    #print("Convolved: " + str(np.arange(tot_data).shape))
    ax.plot(np.arange(tot_data)[lower_boundary:upper_boundary], data_convolved_array, color, alpha=1.0)  # Convolved plot
    fig.savefig(filepath)
    fig.clear()
    plt.close(fig)
    # print(plt.get_fignums())  # print the number of figures opened in background


def plot_curves(data_lists, alphas, filepath="./my_plot.png", x_label="X", y_label="Y", color="-r", kernel_size=50, grid=True):
    """Plot a graph using matplotlib

    """
    if(len(data_lists[0]) <=1):
        print("[WARNING] the data list is empty, no plot will be saved.")
        return
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=True)
    ax.grid(grid)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    #ax.plot(data_list, color, alpha=alpha)  # The original data is showed in background
    kernel = np.ones(int(kernel_size))/float(kernel_size)  # Smooth the graph using a convolution
    for i in range(len(data_lists)):
        data_list = data_lists[i]
        tot_data = len(data_list)
        lower_boundary = int(kernel_size/2.0)
        upper_boundary = int(tot_data-(kernel_size/2.0))
        data_convolved_array = np.convolve(data_list, kernel, 'same')[lower_boundary:upper_boundary]
        #print("arange: " + str(np.arange(tot_data)[lower_boundary:upper_boundary]))
        #print("Convolved: " + str(np.arange(tot_data).shape))
        ax.plot(np.arange(tot_data)[lower_boundary:upper_boundary], data_convolved_array, label = "Alpha: " + str(alphas[i]), alpha=1.0)  # Convolved plot
    ax.legend()
    fig.savefig(filepath)
    fig.clear()
    plt.close(fig)
    # print(plt.get_fignums())  # print the number of figures opened in background
# sarsaFile = 'Sarsa_policy.pickle'
# tdFile = 'TD_policy.pickle'
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# plotActionPolicy(sarsaFile, ax)
# plotStatePolicy(tdFile, ax)
# plt.show()