import numpy as np
import gym
from agent import *
from config import *


def train(agent, env, n_episode, max_t, scale=1):
    reward_log = []
    average_reward = []

    for i in range(n_episode):
        state = env.reset()
        done = False
        t = 0
        state_history = [list(state)]
        action_history = []
        reward_history = []
        dones_history = []
        episodic_reward = 0


        while not done and t <= max_t:
            action = agent.act(state)
            next_sate, reward, done, _ = env.step(action)
            episodic_reward += reward
            action_history.append(action)
            dones_history.append(done)
            reward_history.append(reward * scale)
            state = next_sate
            state_history.append(list(state))
            t += 1

        state_values, log_probs, rewards, discounted_reward, dones = agent.process_data(state_history, action_history, reward_history, dones_history, 64)
        agent.learn(state_values, log_probs, rewards, dones)

        reward_log.append(episodic_reward)
        average_reward.append(np.mean(reward_log[-100:]))
        print('\rEpisode {} Reward {:.2f}, Average Reward {:.2f}'.format(i, episodic_reward, average_reward[-1]),
              end='')
        if i % 100 == 0:
            print()

    return rewards, average_reward


if __name__ == "__main__":
    env = gym.make(RAM_ENV_NAME)
    agent = Agent(env.observation_space.shape[0], env.action_space.n, LEARNING_RATE, GAMMA, DEVICE, SHARE, MODE, CRITIC,
                  NORMALIZE)
    rewards_log, _ = train(agent, env, RAM_NUM_EPISODE, MAX_T, SCALE)
    np.save('{}rewards.npy'.format(RAM_ENV_NAME), rewards_log)