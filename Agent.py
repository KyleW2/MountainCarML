from TDLambda import TDLambda
from Sarsa import Sarsa

if __name__ == "__main__":
    agent = Sarsa(0.05, 0.7, 0.0, render = False, pickle = True, pickleFile = "Sarsa_policy.pickle", load = True)

    try:
        agent.runSeries(1000000)
        agent.close()
    except KeyboardInterrupt:
        agent.savePolicy()
        print("Closed!")