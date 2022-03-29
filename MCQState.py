import random

class MCQState:
    def __init__(self, position: float, velocity: float) -> None:
        self.position = position
        self.velocity = velocity

        self.returns = {0: [], 2: []}
        self.sumOfReturns = {0: 0, 2: 0}

        self.actionValues = {0: 0, 2: 0}
    
    def __str__(self) -> str:
        return "(" + str(self.position) + ", " + str(self.velocity) + ")"
    
    def getMaxAction(self) -> int:
        max = -100000000
        action = -1

        for k, v in self.actionValues.items():
            if v > max:
                max = v
                action = k
        
        return action
    
    def updateReturns(self, action: int, reward: float) -> None:
        self.returns[action].append(reward)
        # Keep a running sum of the returns of avoid a for loop later
        self.sumOfReturns[action] += reward
    
    def getActionValue(self, action: int) -> float:
        return self.actionValues[action]
    
    def updateValue(self, action: int, alpha: float) -> None:
        '''
        # NOT a += just =
        self.actionValues[action] = self.sumOfReturns[action] / len(self.returns[action])
        '''
        # Chaning to profs update funtion
        g = self.sumOfReturns[action] / len(self.returns[action])
        self.actionValues[action] += alpha * (g - self.actionValues[action])