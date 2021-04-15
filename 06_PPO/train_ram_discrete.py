import numpy as np
import gym
from config_discrete import *
from agent import *


def train(agent, env, n_episode, n_updata=4, updata_frequncy=1, max_t=1500, scale=1):
    rewards_log = []
    average_log = []
    states_history = []
    actions_history = []
    dones_history = []
    rewards_history = []

    for i in range(1, n_episode + 1):
        state = env.reset()
        done = False
        t = 0
        if len(states_history) == 0:
            states_history.append(state)
        else:
            states_history[-1] = list(state)

        episodic_reward = 0

        while not done and t < max_t:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            episodic_reward += reward
            actions_history.append(action)
            rewards_history.append(reward * scale)
            dones_history.append(done)
            state = next_state
            states_history.append(list(state))

        if i % updata_frequncy == 0:
            states, actions, log_probs, rewards, dones = agent.process_data(states_history, actions_history,
                                                                            rewards_history, dones_history, 64)
            for _ in range(n_updata):
                agent.learn(states, actions, log_probs, rewards, dones)

            states_history = []
            actions_history = []
            rewards_history = []
            dones_history = []

        rewards_log.append(episodic_reward)
        average_log.append(np.mean(rewards_log[-100:]))

        print('\rEpisode {} Reward {:.2f}, Average Reward {:.2f}'.format(i, episodic_reward, average_log[-1]), end='')
        if not done:
            print('\nEpisode {} did not end'.format(i))
        if i % 200 == 0:
            print()

    return rewards_log, average_log


if __name__ == "__main__":
    env = gym.make(RAM_DISCRETE_ENV_NAME)
    agent = Agent_Discrete(state_size=env.observation_space.shape[0],
                           action_size=env.action_space.n,
                           lr=LEARNING_RATE,
                           beta=BETA,
                           eps=EPS,
                           lambda1=LAMBDA1,
                           gamma=GAMMA,
                           device=DEVICE,
                           hidden=HIDDEN_DISCRETE,
                           share=SHARE,
                           mode=MODE,
                           use_critic=CRITIC,
                           normalize=NORMALIZE)
    reward_log, _ = train(agent=agent,
                          env=env,
                          n_episode=RAM_NUM_EPISODE,
                          n_updata=N_UPDATE,
                          updata_frequncy=UPDATE_FREQUENCY,
                          max_t=MAX_T,
                          scale=SCALE)
    np.save('{}_rewards.npy'.format(RAM_DISCRETE_ENV_NAME), reward_log)
