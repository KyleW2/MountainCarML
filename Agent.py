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
        saveResults((alphas, rewardLists), "SarsaResults.pickle")
    except KeyboardInterrupt:
        saveResults((alphas,rewardLists), "SarsaResultsInterupt.pickle")
        print("Closed!")

def runMCQ(alphas, gamma, epsilon, iters):
    agents = []
    rewardLists = []
    try:
        for i in range(len(alphas)):
            agents.append(MonteCarloQ(alphas[i], 10, epsilon, False, None, False))
            agents[i].runSeries(iters)
            rewardLists.append(agents[i].rewards)
        saveResults((alphas, rewardLists), "QResults.pickle")
    except KeyboardInterrupt:
        saveResults((alphas,rewardLists), "QResultsInterupt.pickle")
        print("Closed!")

def runQ(alphas, gamma, epsilon, iters):
    agents = []
    rewardLists = []
    try:
        for i in range(len(alphas)):
            agents.append(QLearning(alphas[i], gamma, epsilon, False, None, False))
            agents[i].runSeries(iters)
            rewardLists.append(agents[i].rewards)
        saveResults((alphas, rewardLists), "MCQResults.pickle")
    except KeyboardInterrupt:
        saveResults((alphas,rewardLists), "MCQResultsInterupt.pickle")
        print("Closed!")

def plotRewards(rewardLists):
    plot.plot_curves(rewardLists, alphas, filepath="./Sarsa3.png", x_label="Episode", y_label="Reward", color="red", kernel_size=500, grid=True)

if __name__ == "__main__":
    iters = 100000
    gamma = .999
    alphas = [.001, .01, .1, .5, .9]
    epsilon = 1
    runSarsas(alphas, gamma, epsilon, iters)
    runQ(alphas, gamma, epsilon, iters)
    runMCQ(alphas, gamma, epsilon, iters)