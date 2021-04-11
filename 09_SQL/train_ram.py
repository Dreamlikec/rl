import gym
import numpy as np
from config import *
from agent import Agent


def train(env, agent, num_eposide, max_t):
    rewards_log = []
    average_rewards_log = []

    for i in range(1, num_eposide+1):
        state = env.reset()
        done = False
        t = 0
        episodic_reward = 0

        while not done and t < max_t:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.memory.append((state, action, reward, next_state, done))
            t += 1

            if t % 4 == 0 and len(agent.memory) > agent.bs:
                agent.learn()

            state = next_state.copy()
            episodic_reward += reward

        rewards_log.append(episodic_reward)
        average_rewards_log.append(np.mean(rewards_log[-100:]))

        print("\rEpisode {}, Reward {:.2f}, Average Reward {:.2f}".format(i, rewards_log[-1], average_rewards_log[-1]),
              end="")
        if i % 200 == 0:
            print()


if __name__ == "__main__":
    env = gym.make(RAM_ENV_NAME)
    agent = Agent(state_size=env.observation_space.shape[0],
                  action_size=env.action_space.n,
                  alpha=ALPHA,
                  bs=BATCH_SIZE,
                  lr=LEARNING_RATE,
                  tau=TAU,
                  gamma=GAMMA,
                  device=DEVICE)
    train(env, agent, RAM_NUM_EPISODE, MAX_T)
