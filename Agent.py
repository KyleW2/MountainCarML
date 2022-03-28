from TDLambda import TDLambda
from Sarsa import Sarsa
from QLearning import QLearning

if __name__ == "__main__":
    agentSarsa = Sarsa(0.05, 0.7, 0.0, render = False, pickle = True, pickleFile = "Sarsa_policy.pickle", load = False)
    agentTD = TDLambda(lam = 0, alpha = 0.05, epsilon = 0.0, gamma = 0.99,
                     render = False, pickle = True, pickleFile = "TD_policy.pickle", load = False)
    agentQ = QLearning(0.05, 0.7, 1.0, render = False, pickle = True, pickleFile = "Q_policy.pickle", load = False)
    try:
        agentQ.runSeries(100000)
    except KeyboardInterrupt:
        agentQ.savePolicy()
        print("Closed!")