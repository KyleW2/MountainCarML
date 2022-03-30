from MonteCarloQ import MonteCarloQ
from TDLambda import TDLambda
from Sarsa import Sarsa
from QLearning import QLearning
from MonteCarlo import MonteCarlo
from MonteCarloQ import MonteCarloQ
import plot
import pickle as pickler
def saveResults(results, pickleFile) -> None:
    f = open(pickleFile, "wb")
    pickler.dump(results, f)
    f.close()

if __name__ == "__main__":
    # explore = how many steps to explore for at the start
    # explore: int, 0 < explore < 200
    #
    #                     alpha, explore, epsilon
    #agentMCQ = MonteCarloQ(0.05, 10, 0.0, render = False, pickle = True, pickleFile = "Policies/MCQ_policy.pickle", load = False)
    iters = 100000
    gamma = .999
    alphas = [.01, .05, .10, .25, .5, .75, .99]
    #alphas = [.01, .05]
    epsilon = 1
    #agentSarsa = Sarsa(alpha, gamma, epsilon, False, None, False)
    sarsas = []
    rewardLists = []
    try:
        for i in range(len(alphas)):
            sarsas.append(Sarsa(alphas[i], gamma, epsilon, False, None, False))
            sarsas[i].runSeries(iters)
            rewardLists.append(sarsas[i].rewards)
        saveResults(rewardLists, "SarsaResults.pickle")
        plot.plot_curves(rewardLists, alphas, filepath="./Sarsa.png", x_label="Episode", y_label="Reward", color="red", kernel_size=500, grid=True)
        
            #agentMCQ.runSeries(100000)
    except KeyboardInterrupt:
        #agentMCQ.savePolicy()
        #agentSarsa.savePolicy()
        saveResults(rewardLists, "SarsaResultsInterupt.pickle")
        print("Closed!")
