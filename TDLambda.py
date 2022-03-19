import gym
import random

from State import State
from Step import Step

class TDLambda:
    def __init__(self, lam: float, alpha: float, gamma: float, epsilon: float):
        self.env = gym.make("MountainCar-v0")
        
        self.lam = lam
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Keys: tuple = (pos, vel)
        # Value: State = State()
        self.policy = {}

        # Class var for induvidual episodes
        # Reset each time
        self.episode = []

        self.wins = 0

    def resetElig(self) -> None:
        for v in self.policy.values():
            v.resetElig()
    
    def getNextAction(self, state: tuple) -> int:
        # If we've seen the state then get the action for highest value state 
        if state in self.policy.keys():
            # Chance to explore or act greedy
            if random.random() < (1 - self.epsilon):
                possibles =  self.policy[state].getNextStates()

                if len(possibles) == 0:
                    return random.choice([0, 1, 2])

                bestAction = 0
                bestValue = -10000

                for k, v in possibles.items():
                    if self.policy[v].getValue() > bestValue:
                        bestValue = self.policy[v].getValue()
                        bestAction = k
                
                return bestAction
            else:
                return random.choice([0, 1, 2])
        # Otherwise add state to the policy and return a random move
        else:
            self.policy[state] = State(state[0], state[1])
            return random.choice([0, 1, 2])
    
    def runEpisode(self, render = False) -> None:
        self.episode = []
        self.resetElig()

        observation = self.env.reset()
        done = False
        eps = 0

        while not done:
            if render: 
                self.env.render()

            pos = round(observation[0], 2)
            vel = round(observation[1], 2)
            stateTuple = (pos, vel)

            action = self.getNextAction(stateTuple)
            observation, reward, done, info = self.env.step(action)

            self.episode.append(Step(stateTuple, action, reward))

            if reward == 0:
                self.wins += 1

            eps += 1
        
        self.policyEval()
        self.updateNextStates()
        #self.printValues()
    
    def printEpisode(self) -> None:
        print("Episode length: " + str(len(self.episode)))

        for i in range(0, len(self.episode)):
            print(self.episode[i])
    
    def policyEval(self) -> None:
        for i in range(0, len(self.episode) - 1):
            delta = self.episode[i].getReward()
            delta += (self.gamma * self.policy[self.episode[i+1].getState()].getValue()) 
            delta -= self.policy[self.episode[i].getState()].getValue()
            
            self.policy[self.episode[i].getState()].updateElig(self.gamma, self.lam, True)

            for v in self.policy.values():
                v.updateValuePreDelta(self.alpha, delta)
                v.updateElig(self.gamma, self.lam, False)
    
    def updateNextStates(self) -> None:
        for i in range(0, len(self.episode) - 1):
            self.policy[self.episode[i].getState()].addNextState(self.episode[i].getAction(), self.episode[i+1].getState())
    
    def highestPoint(self) -> float:
        max = -1.0
        for i in range(0, len(self.episode)):
            if self.episode[i].getState()[0] > max:
                max = self.episode[i].getState()[0]
        
        return round(max, 4)

    def runSeries(self, episodes: int, render = False) -> None:
        for i in range(0, episodes):
            self.runEpisode(render)
            print(f"Episode {i}, visited {len(self.policy.keys())} states, total wins are {self.wins}")
    
    def close(self):
        self.env.close()