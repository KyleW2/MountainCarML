import gym
import random

from State import State
from Step import Step

class TDLambda:
    def __init__(self, lam: float, alpha: float, gamma: float):
        self.env = gym.make("MountainCar-v0")
        
        self.lam = lam
        self.alpha = alpha
        self.gamma = gamma

        # Keys: tuple = (pos, vel)
        # Value: State = State()
        self.policy = {}

        # Class var for induvidual episodes
        # Reset each time
        self.episode = []

    def resetElig(self) -> None:
        for v in self.policy.values():
            v.resetElig()
    
    def getNextAction(self, state: tuple) -> int:
        # If we've seen the state then get the action for highest value state 
        if state in self.policy.keys():
            possibles =  self.policy[state].getNextStates()
            bestAction = 0
            bestValue = self.policy[possibles[0]].getValue()

            for k, v in possibles.items():
                if self.policy[v].getValue() > bestValue:
                    bestValue = self.policy[v].getValue()
                    bestAction = k
            
            return bestAction
        # Otherwise add state to the policy and return a random move
        else:
            self.policy[state] = State(state[0], state[1])
            return random.choice([0, 1, 2])
    
    def runEpisode(self) -> None:
        self.episode = []
        self.resetElig()

        observation = self.env.reset()
        done = False

        while not done:
            self.env.render()

            stateTuple = (observation[0], observation[1])
            action = self.getNextAction(stateTuple)
            observation, reward, done, info = self.env.step(action)

            self.episode.append(Step(stateTuple, action, reward))
        
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
            self.policy[self.episode[i].getState()].addNextState(self.episode[i].getAction(), self.episode[i].getState())

    def runSeries(self, episodes: int) -> None:
        for i in range(0, episodes):
            self.runEpisode()
            
    def printValues(self):
        for k, v in self.policy.items():
            print("V(" + str(k) + ") = " + str(v.getValue()))
            print("E(" + str(k) + ") = " + str(v.getElig()))
            print()
    
    def close(self):
        self.env.close()

if __name__ == "__main__":
    agent = TDLambda(0.6, 0.05, 0.9)

    try:
        agent.runSeries(100)
        agent.close()
    except KeyboardInterrupt:
        print("Closed!")