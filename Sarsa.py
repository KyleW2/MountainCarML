import gym
import random
import pickle as pickler

from QState import QState
from Step import Step
import numpy as np
import matplotlib.pyplot as plt
class Sarsa:
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
        self.rewards = []
        
    def eGreedy(self, state: tuple):
        # Chance to be greedy
        if random.random() < (1 - self.epsilon):
            # Return max action
            return self.policy[state].getMaxAction()
        # Explore with random action
        else:
            return random.choice([0, 1, 2])
    
    def updateQ(self, stateTuple: tuple, action: int, reward: int, newState: tuple, newAction: int):
        currentQ = self.policy[stateTuple].getActionValue(action)
        nextQ = self.policy[newState].getActionValue(newAction)
        newValue = self.alpha * (reward + (self.gamma * nextQ) - currentQ)

        self.policy[stateTuple].updateActionValue(action, newValue)
        return newValue - currentQ
    
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
            action = self.eGreedy(stateTuple)

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

            # Choose a' from s'
            newAction = self.eGreedy(newState)

            # Update Q(S, A) using s' and a'
            qChange = self.updateQ(stateTuple, action, reward, newState, newAction)
            #print("Q Change is :",qChange)
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
        self.rewards.append(cumulative_reward)

    def runSeries(self, episodes: int) -> None:
        for i in range(0, episodes):
            self.runEpisode()
            if self.epsilon > .1: self.epsilon = .9999*self.epsilon
            print(f"episode: {i}, visited: {len(self.policy.keys())}, wins: {self.wins}, win rate: {self.wins/(i+1)}, epsilon: {self.epsilon}")
        
        self.savePolicy()
    
    def savePolicy(self) -> None:
        if self.pickle:
            f = open(self.pickleFile, "wb")
            pickler.dump(self.policy, f)
            f.close()
    
    def close(self):
        self.env.close()

