from math import inf
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
        saveResults((alphas, rewardLists), "FinalMCQResults.pickle")
        return rewardLists
    except KeyboardInterrupt:
        saveResults((alphas,rewardLists), "FinalMCQResultsInterupt.pickle")
        print("Closed!")

def runQ(alphas, gamma, epsilon, iters):
    agents = []
    rewardLists = []
    try:
        for i in range(len(alphas)):
            agents.append(QLearning(alphas[i], gamma, epsilon, False, None, False))
            agents[i].runSeries(iters)
            rewardLists.append(agents[i].rewards)
        saveResults((alphas, rewardLists), "FinalQResults.pickle")
        return rewardLists
    except KeyboardInterrupt:
        saveResults((alphas,rewardLists), "FinalQResultsInterupt.pickle")
        print("Closed!")

def plotRewards(rewardLists, alphas, filepath):
    plot.plot_curves(rewardLists, alphas, filepath=filepath, x_label="Episode", y_label="Reward", color="red", kernel_size=500, grid=True)

def plotHighestRewards(rewardLists, names, filepath):
    plot.plot_highest_curves(rewardLists, names, filepath=filepath, x_label="Episode", y_label="Reward", kernel_size=500, grid=True)

def plotHighest():
    sarsaFile = "FinalSarsaResults.pickle"
    mcqFile = "FinalMCQResults.pickle"
    qFile = "FinalQResults.pickle"
    
    try:
        fSarsa = open(sarsaFile, "rb")
        fMcq = open(mcqFile, "rb")
        fQ = open(qFile, "rb")
        sarsaRewards = pickler.load(fSarsa)
        mcqRewards = pickler.load(fMcq)
        qRewards = pickler.load(fQ)
    except FileNotFoundError:
        fSarsa.close()
        fMcq.close()
        fQ.close()

    #plotRewards(mcqRewards[1], mcqRewards[0], "test.png")

    highestSarsa = highestReward(sarsaRewards)
    highestMcq = highestReward(mcqRewards)
    highestQ = highestReward(qRewards)
    rewardLists = [highestSarsa, highestMcq, highestQ]
    names = ["Sarsa", "MCQ", "Q"]
    plotHighestRewards(rewardLists, names, "highest.png")

    fSarsa.close()
    fMcq.close()
    fQ.close()

def getWins(rewards):
    wins = []
    for reward in rewards:
        if reward > -200: wins.append(reward)
    return wins

def bestScore():
    sarsaFile = "FinalSarsaResults.pickle"
    mcqFile = "FinalMCQResults.pickle"
    qFile = "FinalQResults.pickle"
    
    try:
        fSarsa = open(sarsaFile, "rb")
        fMcq = open(mcqFile, "rb")
        fQ = open(qFile, "rb")
        sarsaRewards = pickler.load(fSarsa)
        mcqRewards = pickler.load(fMcq)
        qRewards = pickler.load(fQ)
    except FileNotFoundError:
        fSarsa.close()
        fMcq.close()
        fQ.close()

    bestScore = -200
    a = sarsaRewards[1]
    b = mcqRewards[1]
    c = qRewards[1]
    for alpha in a:
        for reward in alpha:
            if reward > bestScore: bestScore = reward
    for alpha in b:
        for reward in alpha:
            if reward > bestScore: bestScore = reward
    for alpha in c:
        for reward in alpha:
            if reward > bestScore: bestScore = reward

    print("Best Score:", bestScore)

def firstWin():
    sarsaFile = "FinalSarsaResults.pickle"
    mcqFile = "FinalMCQResults.pickle"
    qFile = "FinalQResults.pickle"
    
    try:
        fSarsa = open(sarsaFile, "rb")
        fMcq = open(mcqFile, "rb")
        fQ = open(qFile, "rb")
        sarsaRewards = pickler.load(fSarsa)
        mcqRewards = pickler.load(fMcq)
        qRewards = pickler.load(fQ)
    except FileNotFoundError:
        fSarsa.close()
        fMcq.close()
        fQ.close()

    firstSarsa = 100000
    alphaSarsa = None
    firstMCQ = 100000
    alphaMCQ = None
    firstQ = 100000
    alphaQ = None

    print("First Sarsa Win")

    for i in range(len(sarsaRewards[0])):
        rewards = sarsaRewards[1][i]
        alpha = sarsaRewards[0][i]
        for episode in range(len(rewards)):
            if rewards[episode] > -200:
                if firstSarsa > episode: 
                    firstSarsa = episode
                    alphaSarsa = alpha
                break
    print("Episode:",firstSarsa)
    print("Alpha:",alphaSarsa)

    print("First MCQ Win")

    for i in range(len(mcqRewards[0])):
        rewards = mcqRewards[1][i]
        alpha = mcqRewards[0][i]
        for episode in range(len(rewards)):
            if rewards[episode] > -200:
                if firstMCQ > episode: 
                    firstMCQ= episode
                    alphaMCQ = alpha
                break
    print("Episode:",firstMCQ)
    print("Alpha:",alphaMCQ)

    print("First Q Win")

    for i in range(len(qRewards[0])):
        rewards = qRewards[1][i]
        alpha = qRewards[0][i]
        for episode in range(len(rewards)):
            if rewards[episode] > -200:
                if firstQ > episode: 
                    firstQ = episode
                    alphaQ = alpha
                break
    print("Episode:",firstQ)
    print("Alpha:",alphaQ)

def plotWins():
    sarsaFile = "FinalSarsaResults.pickle"
    mcqFile = "FinalMCQResults.pickle"
    qFile = "FinalQResults.pickle"
    
    try:
        fSarsa = open(sarsaFile, "rb")
        fMcq = open(mcqFile, "rb")
        fQ = open(qFile, "rb")
        sarsaRewards = pickler.load(fSarsa)
        mcqRewards = pickler.load(fMcq)
        qRewards = pickler.load(fQ)
    except FileNotFoundError:
        fSarsa.close()
        fMcq.close()
        fQ.close()

    #plotRewards(mcqRewards[1], mcqRewards[0], "test.png")

    highestSarsa = highestReward(sarsaRewards)
    highestMcq = highestReward(mcqRewards)
    highestQ = highestReward(qRewards)
    winsSarsa = getWins(highestSarsa)
    winsMcq = getWins(highestMcq)
    winsQ = getWins(highestQ)
    rewardLists = [winsSarsa, winsMcq, winsQ]
    names = ["Sarsa", "MCQ", "Q"]
    plotHighestRewards(rewardLists, names, "highestWins.png")

    fSarsa.close()
    fMcq.close()
    fQ.close()

def highestReward(rewardList):
    highestSum = -inf
    highestRewards = None
    highestAlpha = None
    for i in range( len(rewardList[1]) ):
        rewards = rewardList[1][i]
        alpha = rewardList[0][i]
        sum = 0
        for reward in rewards:
            sum += reward
        
        if sum > highestSum:
            highestSum = sum
            highestRewards = rewards
            highestAlpha = alpha

    print("Sum:", sum)
    print("Alpha:", highestAlpha)
    return highestRewards

def runExperiments():
    iters = 100000
    gamma = .999
    alphas = [.001, .01, .1, .5, .9]
    epsilon = 1
    sarsaRewards = runSarsas(alphas, gamma, epsilon, iters)
    qRewards = runQ(alphas, gamma, epsilon, iters)
    mcqRewards = runMCQ(alphas, gamma, epsilon, iters)
    plotRewards(sarsaRewards, alphas, "FinalSarsa.png")
    plotRewards(qRewards, alphas, "FinalQ.png")
    plotRewards(mcqRewards, alphas, "FinalMCQ.png")

if __name__ == "__main__":
    bestScore()