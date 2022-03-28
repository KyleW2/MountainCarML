import random

class MCState:
    def __init__(self, position: float, velocity: float) -> None:
        self.position = position
        self.velocity = velocity

        self.returns = []
        self.sumOfReturns = 0
        self.value = 0.0

        self.bestAction = -1
        self.bestValue = 0
    
    def __str__(self) -> str:
        return "(" + str(self.position) + ", " + str(self.velocity) + ")"
    
    def updateBestAction(self, a: int, value: tuple) -> None:
        if value > self.bestValue:
            self.bestAction = a
            self.bestValue = value
    
    def getBestAction(self) -> int:
        if self.bestAction == -1:
            return random.choice([0, 1, 2])
            
        return self.bestAction
    
    def updateReturns(self, reward: float) -> None:
        self.returns.append(reward)
        # Keep a running sum of the returns of avoid a for loop later
        self.sumOfReturns += reward
    
    def getValue(self) -> float:
        return self.value
    
    def updateValue(self) -> None:
        # NOT a += just =
        self.value = self.sumOfReturns / len(self.returns)