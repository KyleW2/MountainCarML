from MonteCarloQ import MonteCarloQ
from TDLambda import TDLambda
from Sarsa import Sarsa
from QLearning import QLearning
from MonteCarlo import MonteCarlo
from MonteCarloQ import MonteCarloQ

if __name__ == "__main__":
    # explore = how many steps to explore for at the start
    # explore: int, 0 < explore < 200
    #
    #                     alpha, explore, epsilon
    agentMCQ = MonteCarloQ(0.05, 10, 0.0, render = False, pickle = True, pickleFile = "Policies/MCQ_policy.pickle", load = False)

    try:
        agentMCQ.runSeries(100000)

    except KeyboardInterrupt:
        agentMCQ.savePolicy()
        print("Closed!")
