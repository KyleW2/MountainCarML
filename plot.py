import pickle
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
f = open('Sarsa_policy.pickle', "rb")
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
ax.scatter(x, y, z)
plt.show()