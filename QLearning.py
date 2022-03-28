import gym
import random
import pickle as pickler
import numpy as np
import matplotlib.pyplot as plt
from QState import QState
from Step import Step

class QLearning:
    def __init__(self, alpha: float, gamma: float, epsilon: float, render = False, pickle = False, pickleFile = None, load = False) -> None:
        self.env = gym.make("MountainCar-v0")
        
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Keys: tuple = (pos, vel)
        # Value: State = State()
        self.policy = {}

        self.render = render
        self.pickle = pickle

        # Pickle stuff
        if self.pickle:
            self.pickleFile = pickleFile

            try:
                f = open(self.pickleFile, "rb")
            except FileNotFoundError:
                f = open(self.pickleFile, "w")
                f.close()
                f = open(self.pickleFile, "rb")

            if load:
                self.policy = pickler.load(f)
            f.close()

        # Class var for induvidual episodes
        # Reset each time
        self.episode = []

        # Metrics
        self.wins = 0
        self.highestPoint = 0
    
    def eGreey(self, state: tuple):
        # Chance to be greedy
        if random.random() < (1 - self.epsilon):
            # Return max action
            return self.policy[state].getMaxAction()
        # Explore with random action
        else:
            return random.choice([0, 1, 2])
    
    def updateQ(self, stateTuple: tuple, action: int, reward: int, newState: tuple):
        currentQ = self.policy[stateTuple].getActionValue(action)
        maxA = self.policy[newState].getMaxAction()
        maxNextQ = self.policy[newState].getActionValue(maxA)

        newValue = self.alpha * (reward + (self.gamma * maxNextQ) - currentQ)

        self.policy[stateTuple].updateActionValue(action, newValue)
    
    def runEpisode(self) -> None:
        # Reset the stuff
        self.episode = []
        observation = self.env.reset()
        done = False
        steps = 0
        highest = -1.2
        cumulative_reward = 0
        # Loop for each step
        while not done:
            # Toggle rendering
            if self.render: 
                self.env.render()

            # Create tuple representing current state
            pos = round(observation[0], 1)
            vel = round(observation[1], 2)
            stateTuple = (pos, vel)

            # Add state to policy if new
            if stateTuple not in self.policy.keys():
                self.policy[stateTuple] = QState(pos, vel)
            
            # Find next action using e-greedy
            action = self.eGreey(stateTuple)

            # Observe r and s'
            observation, reward, done, info = self.env.step(action)

            # Make tuple for s' and add to policy if new
            newPos = round(observation[0], 1)
            newVel = round(observation[1], 2)
            newState = (newPos, newVel)

            if newState not in self.policy.keys():
                self.policy[newState] = QState(newPos, newVel)

            # Ensure reward is correct
            if newPos >= 0.5:
                reward = 0
            else:
                reward = -1

            # Update Q(S, A) using s' and a'
            self.updateQ(stateTuple, action, reward, newState)

            # Add step to episode
            self.episode.append(Step(stateTuple, action, reward))

            # Metrics
            if reward == 0 and done:
                self.wins += 1
            if observation[0] > highest:
                highest = observation[0]
            self.highestPoint = highest
            steps += 1
            cumulative_reward += reward
        return cumulative_reward
    def runSeries(self, episodes: int) -> None:
        rewards = list()
        for i in range(0, episodes):
            self.runEpisode()
            rewards.append(self.runEpisode())

            print(f"episode: {i}, visited: {len(self.policy.keys())}, wins: {self.wins}, win rate: {self.wins/(i+1)}, epsilon: {self.epsilon}")
            if self.epsilon > .1: self.epsilon = .9999*self.epsilon
        self.savePolicy()
        QLearning.plot_curve(rewards, filepath="./Qreward.png",
            x_label="Episode", y_label="Reward",
            x_range=(0, len(rewards)), y_range=(-3.1,0.1),
            color="red", kernel_size=500,
            alpha=0.4, grid=True)
    
    def savePolicy(self) -> None:
        if self.pickle:
            f = open(self.pickleFile, "wb")
            pickler.dump(self.policy, f)
            f.close()
    
    def close(self):
        self.env.close()

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