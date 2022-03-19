
class State:
    def __init__(self, position: float, velocity: float) -> None:
        self.position = position
        self.velocity = velocity
        self.e = 0.0
        self.value = 0.0

        # Keys: int = 0, 1, 2 (the move that lead to the state)
        # Values: tuple = state tuple
        self.nextStates = {}
    
    def __str__(self) -> str:
        return "(" + str(self.position) + ", " + str(self.velocity) + ")"
    
    def addNextState(self, a: int, s: tuple) -> None:
        if a in self.nextStates.keys() and not s == self.nextStates[a]:
            print("Over-writing a states list of next states!!!")
        self.nextStates[a] = s
    
    def getNextStates(self) -> dict:
        return self.nextStates
    
    def getValue(self) -> float:
        return self.value
    
    def getElig(self) -> float:
        return self.e
    
    def updateValue(self, alpha: float, gamma: float, reward: float, next_value: float) -> None:
        delta = reward + (gamma * next_value) - self.value
        self.value = self.value + (alpha * delta * self.e)

    def updateValuePreDelta(self, alpha: float, delta: float) -> None:
        self.value = self.value + (alpha * delta * self.e)
    
    def updateElig(self, gamma: float, lam: float, is_state: bool) -> None:
        if is_state:
            self.e += 1
        else:
            self.e = gamma * lam * self.e
    
    def resetElig(self) -> None:
        self.e = 0