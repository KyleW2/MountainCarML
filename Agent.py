from MonteCarloQ import MonteCarloQ
from TDLambda import TDLambda
from Sarsa import Sarsa
from QLearning import QLearning
from MonteCarlo import MonteCarlo
from MonteCarloQ import MonteCarloQ

if __name__ == "__main__":
    agentMC = MonteCarloQ(0.05, 0.999, 0.0, render = False, pickle = True, pickleFile = "MC_policy.pickle", load = True)

    try:
        agentMC.runSeries(100000)

    except KeyboardInterrupt:
        agentMC.savePolicy()
        print("Closed!")
