import gym
import random
import numpy as np
from agent import SAC_Agent
from config import *


def train(env, agent, n_episode, max_t):
    rewards_log = []
    average_rewards = []

    for i in range(1, n_episode + 1):
        state = env.reset()
        episodic_reward = 0
        done = False
        t = 0

        while not done and t < max_t:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            scale = SCALE
            agent.memory.add(state, action, scale * reward, next_state, done)
            t += 1

            if len(agent.memory) > agent.batch_size:
                for _ in range(agent.update_frequency):
                    agent.learn()
            state = next_state.copy()
            episodic_reward += reward

        rewards_log.append(episodic_reward)
        average_rewards.append(np.mean(rewards_log[-100:]))

        print("\rEpisode {}, Reward {:.2f}, Average Rewards {:.2f}".format(i, rewards_log[-1], average_rewards[-1]),
              end="")
        if i % 100 == 0:
            print()


if __name__ == "__main__":
    env = gym.make(RAM_ENV_NAME)
    agent = SAC_Agent(env=env,
                      device=DEVICE,
                      N=N,
                      lr=LEARNING_RATE,
                      alpha=ALPHA,
                      std=STD,
                      update_frequency=UPDATE_FREQUENCY,
                      tau=TAU,
                      buffer_size=BUFFER_SIZE,
                      batch_size=BATCH_SIZE,
                      log_in_V_loss=LOG_IN_V,
                      log_in_pi_loss=LOG_IN_PI)
    train(env, agent, RAM_NUM_EPISODE, MAX_T)
