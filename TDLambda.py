import gym
import random
import pickle as pickler

from State import State
from Step import Step

class TDLambda:
    def __init__(self, lam: float, alpha: float, gamma: float, epsilon: float, render = False, pickle = False, pickleFile = None, load = False):
        self.env = gym.make("MountainCar-v0")
        
        self.lam = lam
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.render = render

        # Keys: tuple = (pos, vel)
        # Value: State = State()
        self.policy = {}
        self.pickle = pickle

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

        self.wins = 0
        self.highestPoint = 0

    def resetElig(self) -> None:
        for v in self.policy.values():
            v.resetElig()
    
    def eGreedy(self, state: tuple) -> int:
        # Chance to explore or act greedy
        if random.random() < (1 - self.epsilon):
            return self.policy[state].getBestAction()
        else:
            return random.choice([0, 1, 2])
    
    def runEpisode(self) -> None:
        # Reset stuff that needs to be reset
        self.episode = []
        self.resetElig()
        observation = self.env.reset()
        done = False
        step = 0
        highest = -1.2

        # Loop for each step
        while not done:
            # Toggle rendering
            if self.render: 
                self.env.render()

            # Create tuple representing current state
            pos = round(observation[0], 2)
            vel = round(observation[1], 2)
            stateTuple = (pos, vel)

            # Add state to policy if new
            if stateTuple not in self.policy.keys():
                self.policy[stateTuple] = State(pos, vel)

            # Find next action using e-greedy policy
            action = self.eGreedy(stateTuple)

            # Take action a and observe r, s'
            observation, reward, done, info = self.env.step(action)

            # Update state with a and s'
            newState = (round(observation[0], 2), round(observation[1], 2))

            # Add s' to policy if new
            if newState not in self.policy.keys():
                self.policy[newState] = State(pos, vel)

            self.updateBestAction(stateTuple, action, newState)

            # Add Step object to episode
            self.episode.append(Step(stateTuple, action, reward))

            # Update values and eligiblity traces for all s
            self.updateStateValues(reward, newState, stateTuple)

            # Metrics
            if reward == 0:
                self.wins += 1
            if observation[0] > highest:
                highest = observation[0]
            self.highestPoint = highest
            step += 1
    
    def updateStateValues(self, reward, newState, currentState) -> None:
        # Calculate delta
        delta = reward
        delta += (self.gamma * self.policy[newState].getValue()) 
        delta -= self.policy[currentState].getValue()

        # Update eligiblity trace for current state
        self.policy[currentState].updateElig(self.gamma, self.lam, True)

        # Loop thru all states and update
        for s in self.policy.values():
            s.updateValuePreDelta(self.alpha, delta)
            s.updateElig(self.gamma, self.lam, False)
    
    def updateBestAction(self, stateTuple: tuple, action: int, nextState: tuple) -> None:
        nextStateValue = self.policy[nextState].getValue()
        self.policy[stateTuple].updateBestAction(action, nextStateValue)

    def runSeries(self, episodes: int) -> None:
        for i in range(0, episodes):
            self.runEpisode()
            print(f"episode: {i}, visited: {len(self.policy.keys())}, wins: {self.wins}, highest: {self.highestPoint}")
        
        self.savePolicy()
    
    def savePolicy(self) -> None:
        if self.pickle:
            f = open(self.pickleFile, "wb")
            pickler.dump(self.policy, f)
            f.close()
    
    def close(self):
        self.env.close()