from TDLambda import TDLambda
from Sarsa import Sarsa

if __name__ == "__main__":
    agent = Sarsa(0.5, 0.5, 0.0, render = False, pickle = True, pickleFile = "Sarsa_policy.pickle", load = False)

    try:
        agent.runSeries(100000)
        agent.close()
    except KeyboardInterrupt:
        agent.savePolicy()
        print("Closed!")