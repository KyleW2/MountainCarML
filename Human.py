import gym

env = gym.make("MountainCar-v0")
print(env.action_space)
print(env.observation_space)

for i_episode in range(0, 20):
    observation = env.reset()
    for t in range(0, 1000):

        env.render()
        print(observation)

        if t % 2 == 0:
            action = 0
        else:
            action = 0
        
        if t > 500:
            action = 1

        observation, reward, done, info = env.step(action)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.close()