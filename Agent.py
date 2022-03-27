from TDLambda import TDLambda
from Sarsa import Sarsa

if __name__ == "__main__":
    agentSarsa = Sarsa(0.05, 0.7, 0.0, render = False, pickle = True, pickleFile = "Sarsa_policy.pickle", load = False)
    agentTD = TDLambda(lam = 0.7, alpha = 0.05, epsilon = 1.0, gamma = .5,
                     render = False, pickle = True, pickleFile = "TD_policy.pickle", load = False)
    try:
        agentTD.runSeries(10000)
        agentTD.close()
        agentSarsa.runSeries(10000)
        agentSarsa.close()
    except KeyboardInterrupt:
        agentTD.savePolicy()
        agentSarsa.savePolicy()
        print("Closed!")