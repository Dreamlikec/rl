import gym
import numpy as np
from config import *
from agent import Agent


def train(agent: Agent, env, n_episode):
    rewards_log = []
    average_rewards = []

    for i in range(1, 1 + n_episode):
        state = env.reset()
        action = agent.act(state, i)
        done = False
        reward_episode = 0
        t = 0

        while not done:
            next_state, reward, done, _ = env.step(action)
            agent.memory.add(state, action, reward, next_state, done)
            reward_episode += reward
            t += 1
            if len(agent.memory) > agent.batch_size:
                if t % agent.step == 0:
                    agent.learn()

            state = next_state.copy()
            action = agent.act(state, i)

        rewards_log.append(reward_episode)
        average_rewards.append(np.mean(rewards_log[-100:]))

        print("\rEpisode {}, Reward {:.2f}, Average Reward {:.2f}".format(i, rewards_log[-1], average_rewards[-1]),
              end="")
        if i % 200 == 0:
            print()


if __name__ == "__main__":
    env = gym.make(RAM_ENV_NAME)
    agent = Agent(env=env,
                  lr=LR,
                  tau=TAU,
                  device=DEVICE,
                  step=STEP,
                  delay=DELAY,
                  batch_size=BATCH_SIZE,
                  exploration_level=EXPLORATION_LEVEL,
                  smoothing_level=SMOOTHING_LEVEL,
                  smoothing_max=SMOOTHING_MAX,
                  batch_norm=False,
                  clip=True,
                  initialize=True,
                  hidden=[400, 300])
    train(agent, env, 1000)
