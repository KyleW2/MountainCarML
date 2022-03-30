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

def runSarsas(alphas, gamma, epsilon, iters):
    agents = []
    rewardLists = []
    try:
        for i in range(len(alphas)):
            agents.append(Sarsa(alphas[i], gamma, epsilon, False, None, False))
            agents[i].runSeries(iters)
            rewardLists.append(agents[i].rewards)
        saveResults((alphas, rewardLists), "FinalSarsaResults.pickle")
        return rewardLists
    except KeyboardInterrupt:
        saveResults((alphas,rewardLists), "FinalSarsaResultsInterupt.pickle")
        print("Closed!")

def runMCQ(alphas, gamma, epsilon, iters):
    agents = []
    rewardLists = []
    try:
        for i in range(len(alphas)):
            agents.append(MonteCarloQ(alphas[i], 10, epsilon, False, None, False))
            agents[i].runSeries(iters)
            rewardLists.append(agents[i].rewards)
        saveResults((alphas, rewardLists), "FinalQResults.pickle")
        return rewardLists
    except KeyboardInterrupt:
        saveResults((alphas,rewardLists), "FinalQResultsInterupt.pickle")
        print("Closed!")

def runQ(alphas, gamma, epsilon, iters):
    agents = []
    rewardLists = []
    try:
        for i in range(len(alphas)):
            agents.append(QLearning(alphas[i], gamma, epsilon, False, None, False))
            agents[i].runSeries(iters)
            rewardLists.append(agents[i].rewards)
        saveResults((alphas, rewardLists), "FinalMCQResults.pickle")
        return rewardLists
    except KeyboardInterrupt:
        saveResults((alphas,rewardLists), "FinalMCQResultsInterupt.pickle")
        print("Closed!")

def plotRewards(rewardLists, filepath):
    plot.plot_curves(rewardLists, alphas, filepath=filepath, x_label="Episode", y_label="Reward", color="red", kernel_size=500, grid=True)

if __name__ == "__main__":
    iters = 100000
    gamma = .999
    alphas = [.001, .01, .1, .5, .9]
    epsilon = 1
    sarsaRewards = runSarsas(alphas, gamma, epsilon, iters)
    qRewards = runQ(alphas, gamma, epsilon, iters)
    mcqRewards = runMCQ(alphas, gamma, epsilon, iters)
    plotRewards(sarsaRewards, "FinalSarsa.png")
    plotRewards(qRewards, "FinalQ.png")
    plotRewards(mcqRewards, "FinalMCQ.png")