import gym

env = gym.make("LunarLanderContinuous-v2")
env.reset()


num_episode = 5
max_t = 1000
reward_log = []
t_log = []

for _ in range(num_episode):

    # initialize
    env.reset()
    t = 0
    episodic_reward = 0

    for t in range(max_t):
        env.render()

        action = env.action_space.sample()  # random action
        print(action)
        _, reward, done, _ = env.step(action)
        episodic_reward += reward
        if done:
            break

env.close()