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
        z.append(-state.actionValues[state.getMaxAction()])

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
        z.append(-state.getValue())

    #X, Y = np.meshgrid(x, y)
    #Z = f(x,y)
    ax.scatter(x, y, z, edgecolors='red')
    

sarsaFile = 'Sarsa_policy.pickle'
tdFile = 'TD_policy.pickle'
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plotActionPolicy(sarsaFile, ax)
plotStatePolicy(tdFile, ax)
plt.show()