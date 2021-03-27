import gym
import numpy as np
from agent import Agent
from config import *


def train(n_episode, agent, env, oun, max_t=1500):
    reward_log = []
    average_reward_log = []

    for i in range(1, n_episode + 1):
        done = False
        state = env.reset()
        action = agent.act(state, i)
        episodic_reward = 0
        t = 0

        if oun:
            agent.noise.reset()

        while not done and t < max_t:
            next_state, reward, done, _ = env.step(action)
            agent.memory.add(state, action, reward, next_state, done)
            t += 1

            if len(agent.memory.memory) > agent.batch_size:
                if t % agent.step == 0:
                    for _ in range(agent.learning_time):
                        agent.learn()

            state = next_state.copy()
            action = agent.act(state, i)
            episodic_reward += reward

        reward_log.append(episodic_reward)
        average_reward_log.append(np.mean(reward_log[-100:]))

        print('\rEpisode {} ,REWARD {:.2f} ,AVERAGE REWARD {:.2f}'.format(i, reward_log[-1], average_reward_log[-1]),
              end='')
        if i % 200 == 0:
            print()
    return reward_log, average_reward_log


if __name__ == "__main__":
    env = gym.make(RAM_ENV_NAME)
    agent = Agent(env, lr1=LR1, lr2=LR2, tau=TAU, gamma=GAMMA, step=STEP, speed1=SPEED1, speed2=SPEED2,
                  learning_time=LEARNING_TIME, batch_size=BATCH_SIZE, OUN_noise=OUN, batch_norm=BN, clip=CLIP,
                  initialize=INIT, hidden=HIDDEN)
    train(n_episode=1000, agent=agent, env=env, oun=OUN)