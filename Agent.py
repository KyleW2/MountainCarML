from State import State

class TDLambda:
    def __init__(self, lam: float, alpha: float, gamma: float, states: list):
        self.lam = lam
        self.alpha = alpha
        self.gamma = gamma

        # Parse state objects from list of lists
        self.states = {}
        for i in range(0, len(states)):
            actions = []

            for j in range(0, len(states[i][1])):
                actions.append(states[i][1][j])

            #print(states[i][0], actions)
            self.states[states[i][0]] = State(actions)

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
    
    def runEpisode(self, episode: list) -> None:
        # Episodes must go: [[state -> action -> reward], ...]
        for i in range(0, len(episode)-1):
            delta = episode[i][2] + (self.gamma * self.states[episode[i+1][0]].getValue()) - self.states[episode[i][0]].getValue()
            self.updateElig(episode[i][0], True)

            for k, v in self.states.items():
                # If the state equals current at time step
                self.updateValuePreDelta(k, delta)
                self.updateElig(k, False)
            
    def printValues(self):
        for k, v in self.states.items():
            print("V(" + k + ") = " + str(v.getValue()))
            print("E(" + k + ") = " + str(v.getElig()))

if __name__ == "__main__":
    agent = TDLambda()