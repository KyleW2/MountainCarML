from TDLambda import TDLambda
from Sarsa import Sarsa
from QLearning import QLearning

if __name__ == "__main__":
<<<<<<< HEAD
    agentSarsa = Sarsa(0.05, 0.7, 0.0, render = False, pickle = True, pickleFile = "Sarsa_policy.pickle", load = False)
    agentTD = TDLambda(lam = 0, alpha = 0.05, epsilon = 0.0, gamma = 0.99,
=======
    agentSarsa = Sarsa(0.05, 0.999, 1, render = False, pickle = True, pickleFile = "Sarsa_policy.pickle", load = False)
    agentQ = QLearning(0.05, 0.999, 1, render = False, pickle = True, pickleFile = "Sarsa_policy.pickle", load = False)
    agentTD = TDLambda(lam = 1, alpha = 0.05, epsilon = 1, gamma = .999,
>>>>>>> a4c02dbf6d5409d2aea5b1622fea7bea0bd413fd
                     render = False, pickle = True, pickleFile = "TD_policy.pickle", load = False)
    agentQ = QLearning(0.05, 0.7, 1.0, render = False, pickle = True, pickleFile = "Q_policy.pickle", load = False)
    try:
<<<<<<< HEAD
        agentQ.runSeries(100000)
    except KeyboardInterrupt:
        agentQ.savePolicy()
        print("Closed!")
=======
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
>>>>>>> a4c02dbf6d5409d2aea5b1622fea7bea0bd413fd
