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

    def getValue(self, state: str) -> float:
        return self.states[state].getValue()
    
    def updateValue(self, state: str, reward: int, next_state: str) -> None:
        self.states[state].updateValue(self.alpha, self.gamma, reward, self.states[next_state].getValue())
    
    def updateValuePreDelta(self, state: str, delta: float) -> None:
        self.states[state].updateValuePreDelta(self.alpha, delta)
    
    def updateElig(self, state: str, is_state: bool) -> None:
        self.states[state].updateElig(self.gamma, self.lam, is_state)
    
    def resetElig(self) -> None:
        for k, v in self.states.items():
            v.resetElig()
    
    def getNextAction(self, state: tuple) -> int:
        if state in self.policy.keys():
            return self.policy[state].getNextAction()
        else:
            return random.choice([0, 1, 2])
    
    def runEpisode(self) -> None:
        self.episode = []

        observation = self.env.reset()
        done = False

        while not done:
            self.env.render()

            stateTuple = (observation[0], observation[1])
            action = self.getNextAction(stateTuple)
            observation, reward, done, info = self.env.step(action)

            self.episode.append(Step(stateTuple, action, reward))
        
        self.policyEval()
    
    def policyEval(self) -> None:
        print("Episode length: " + str(len(self.episode)))

        for i in range(0, len(self.episode)):
            print(self.episode[i])
    
    def runSeries(self, episodes: int) -> None:
        for i in range(0, episodes):
            self.runEpisode()
            
    def printValues(self):
        for k, v in self.states.items():
            print("V(" + k + ") = " + str(v.getValue()))
            print("E(" + k + ") = " + str(v.getElig()))
    
    def close(self):
        self.env.close()

if __name__ == "__main__":
    agent = TDLambda(0.5, 0.5, 0.5)
    agent.runSeries(100)
    agent.close()