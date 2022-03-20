import random

class QState:
    def __init__(self, position: float, velocity: float) -> None:
        self.position = position
        self.velocity = velocity

        # Keys: int = action
        # Values: float = value
        self.actionValues = {0: 0, 1: 0, 2: 0}
    
    def getActionValue(self, action: int) -> float:
        return self.actionValues[action]
    
    def getMaxAction(self) -> int:
        max = -100000000
        action = -1

        for k, v in self.actionValues.items():
            if v > max:
                max = v
                action = k
        
        return action
    
    def updateActionValue(self, action: int, value: float) -> None:
        self.actionValues[action] += value