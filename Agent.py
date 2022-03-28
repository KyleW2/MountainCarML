from TDLambda import TDLambda
from Sarsa import Sarsa
from QLearning import QLearning

if __name__ == "__main__":
    agentSarsa = Sarsa(0.05, 0.999, 1, render = False, pickle = True, pickleFile = "Sarsa_policy.pickle", load = False)
    agentQ = QLearning(0.05, 0.999, 1, render = False, pickle = True, pickleFile = "Sarsa_policy.pickle", load = False)
    agentTD = TDLambda(lam = 1, alpha = 0.05, epsilon = 1, gamma = .999,
                     render = False, pickle = True, pickleFile = "TD_policy.pickle", load = False)
    try:
        #agentTD.runSeries(100000)
        #agentTD.close()
        agentQ.runSeries(100000)
        agentQ.close()
        #agentSarsa.runSeries(100000)
        #agentSarsa.close()
    except KeyboardInterrupt:
        #agentTD.savePolicy()
        agentQ.savePolicy()
        #agentSarsa.savePolicy()
        print("Closed!")
