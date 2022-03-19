from TDLambda import TDLambda

if __name__ == "__main__":
    agent = TDLambda(0.6, 0.05, 0.9)

    try:
        agent.runSeries(100000, render = False)
        agent.close()
    except KeyboardInterrupt:
        print("Closed!")