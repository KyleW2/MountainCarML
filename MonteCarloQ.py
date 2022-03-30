import gym
import random
import pickle as pickler
import numpy as np
import matplotlib.pyplot as plt
from MCQState import MCQState
from Step import Step

class MonteCarloQ:
    def __init__(self, alpha: float, explore: float, epsilon: float, render = False, pickle = False, pickleFile = None, load = False) -> None:
        self.env = gym.make("MountainCar-v0")
        
        self.alpha = alpha
        self.explore = explore
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
    def eGreey(self, state: tuple):
        # Chance to be greedy
        if random.random() < (1 - self.epsilon):
            # Return max action
            return self.policy[state].getMaxAction()
        # Explore with random action
        else:
            return random.choice([0, 2])
    
    def randomAction(self) -> int:
        return random.choice([0, 2])
    
    def updateV(self):
        # Nest for loops uh-oh!
        # RIP and sort of speed
        alreadyVisited = []
        for i in range(0, len(self.episode)):
            stateTuple = self.episode[i].getState()
            action = self.episode[i].getAction()

            # If first occurence of s
            if (stateTuple, action) not in alreadyVisited:
                # Sum rewards
                r = 0
                for j in range(i, len(self.episode)):
                    r += self.episode[j].getReward()

                # Append r to s's returns
                self.policy[stateTuple].updateReturns(action, r)
                self.policy[stateTuple].updateValue(action, self.alpha)

                # Added to list of visited states
                alreadyVisited.append((stateTuple, action))
    
    def runEpisode(self) -> None:
        # Reset the stuff
        self.episode = []
        observation = self.env.reset()
        done = False
        step = 0
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
                self.policy[stateTuple] = MCQState(pos, vel)
            
            # Find next action using e-greedy
            action = self.eGreey(stateTuple)

            if step <= self.explore:
                action = self.randomAction()

            # Observe r and s'
            observation, reward, done, info = self.env.step(action)

            # Ensure reward is correct
            if observation[0] >= 0.5:
                reward = 0
            else:
                reward = -1

            # Add step to episode
            self.episode.append(Step(stateTuple, action, reward))

            # Metrics
            if reward == 0 and done:
                self.wins += 1
            if observation[0] > highest:
                highest = observation[0]
            self.highestPoint = highest
            step += 1
            cumulative_reward += reward
        self.updateV()
        self.rewards.append(cumulative_reward)

    def runSeries(self, episodes: int) -> None:
        for i in range(0, episodes):
            self.runEpisode()
            print(f"episode: {i}, visited: {len(self.policy.keys())}, wins: {self.wins}, win rate: {self.wins/(i+1)}, epsilon: {self.epsilon}, highest: {self.highestPoint}")
            if self.epsilon > .1: self.epsilon *= .9999
        self.savePolicy()

    def savePolicy(self) -> None:
        if self.pickle:
            f = open(self.pickleFile, "wb")
            pickler.dump(self.policy, f)
            f.close()
    
    def close(self):
        self.env.close()