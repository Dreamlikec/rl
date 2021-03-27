from config_continuous import *
from agent import *
from config_continuous import *
import gym
import numpy as np


def train(agent, env, n_episode, n_update=4, update_frequnecy=1, max_t=1500, scale=1):
    states_history = []
    actions_history = []
    rewards_history = []
    rewards_log = []
    dones_history = []
    average_rewards_log = []

    for i in range(1, 1 + n_episode):
        state = env.reset()
        done = False
        if len(states_history) == 0:
            states_history.append(list(state))
        else:
            states_history[-1] = list(state)
        t = 0
        episodic_reward = 0
        while not done and t < max_t:
            # action = env.action_space.sample()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            states_history.append(list(state))
            rewards_history.append(scale * reward)
            actions_history.append(action)
            episodic_reward += reward
            dones_history.append(done)

        if i % update_frequnecy == 0:  # begin update
            states, actions, old_mu, old_log_std, old_log_probs, rewards, dones = \
                agent.process_data(states_history, actions_history, rewards_history, dones_history, 64)

            for _ in range(n_update):
                agent.learn(states, actions, old_mu, old_log_std, old_log_probs, rewards, dones)
            states_history = []
            actions_history = []
            rewards_history = []
            dones_history = []

        rewards_log.append(episodic_reward)
        average_rewards_log.append(np.mean(rewards_log[-100:]))

        print('\rEpisode {} Reward {:.2f}, Average Reward {:.2f}'.format(i, episodic_reward, average_rewards_log[-1]),
              end='')
        if not done:
            print('\nEpisode {} did not end'.format(i))
        if i % 200 == 0:
            print()
        torch.cuda.empty_cache()
    return rewards_log, average_rewards_log


if __name__ == "__main__":
    env = gym.make(RAM_CONTINUOUS_ENV_NAME)
    agent = Agent_continuous(state_size=env.observation_space.shape[0],
                             action_size=env.action_space.shape[0],
                             lr=LEARNING_RATE,
                             beta=BETA,
                             lambda1=TAU,
                             gamma=GAMMA,
                             device=DEVICE,
                             hidden=HIDDEN_CONTINUOUS,
                             mode=MODE,
                             use_critic=CRITIC,
                             normalize=NORMALIZE,
                             eps=EPS,
                             share=SHARE)

    reward_log, _ = train(agent=agent, env=env, n_episode=RAM_NUM_EPISODE, update_frequnecy=UPDATE_FREQUENCY,
                          max_t=MAX_T, scale=SCALE)
