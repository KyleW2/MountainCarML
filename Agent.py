from TDLambda import TDLambda
from Sarsa import Sarsa

if __name__ == "__main__":
    agentSarsa = Sarsa(0.05, 0.9, 0.0, render = False, pickle = True, pickleFile = "Sarsa_policy.pickle", load = False)
    agentTD = TDLambda(lam = 1, alpha = 0.05, epsilon = 1, gamma = .99,
                     render = False, pickle = True, pickleFile = "TD_policy.pickle", load = False)
    try:
        agentTD.runSeries(500000)
        agentTD.close()
        agentSarsa.runSeries(20000)
        agentSarsa.close()
    except KeyboardInterrupt:
        agentTD.savePolicy()
        agentSarsa.savePolicy()
        print("Closed!")