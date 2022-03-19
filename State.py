
class State:
    def __init__(self, actions: list, probabilities = None, value = None) -> None:
        self.e = 0
        self.actions = {}

        # Disregard this, only here for future use
        for i in range(0, len(actions)):
            if probabilities != None:
                self.actions[actions[i]] = probabilities[i]
            else:
                self.actions[actions[i]] = 0.0
        
        self.value = 0.0
        if value != None:
            self.value = value
    
    def getValue(self) -> float:
        return self.value
    
    def getElig(self) -> float:
        return self.e
    
    def updateValue(self, alpha, gamma, reward, next_value) -> None:
        delta = reward + (gamma * next_value) - self.value
        self.value = self.value + (alpha * delta * self.e)

    def updateValuePreDelta(self, alpha, delta) -> None:
        self.value = self.value + (alpha * delta * self.e)
    
    def updateElig(self, gamma: float, lam: float, is_state: bool) -> None:
        if is_state:
            self.e += 1
        else:
            self.e = gamma * lam * self.e
    
    def resetElig(self) -> None:
        self.e = 0