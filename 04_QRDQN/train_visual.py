import gym
import numpy as np
from utils import *
from collections import deque
from config import *
from agent import Agent


def train(env, agent, num_episode, eps_init, eps_decay, eps_min, max_t, num_frame=4):
    rewards_log = []
    average_log = []
    eps = eps_init

    for i in range(1, num_episode + 1):
        done = False
        episode_reward = 0
        frame = env.reset()
        frame = gym_preprocess(frame)
        state_deque = deque(maxlen=num_frame)
        for _ in range(num_frame):
            state_deque.append(frame)
        state = np.stack(state_deque, axis=0)
        state = np.expand_dims(state, axis=0)
        t = 0

        while not done and t < max_t:
            t += 1
            action = agent.act(state, eps)
            frame, reward, done, _ = env.step(action)
            frame = gym_preprocess(frame)
            state_deque.append(frame)
            next_state = np.stack(state_deque, axis=0)
            next_state = np.expand_dims(next_state, axis=0)
            transition = (state, action, reward, next_state, done)
            agent.memory.remember(transition)

            if t % 5 == 0 and len(agent.memory) >= agent.batch_size:
                agent.learn()
                agent.soft_update()

            state = next_state.copy()
            episode_reward += reward

        rewards_log.append(episode_reward)
        average_log.append(np.mean(rewards_log[-100:]))
        print("\rEpisode:{}, Reward:{:.3f}, Average:{:.3f}".format(i, episode_reward, average_log[-1]), end='')
        if i % 100 == 0:
            print()

        eps = max(eps_min, eps * eps_decay)
    return rewards_log, average_log


if __name__ == "__main__":
    env = gym.make(VISUAL_ENV_NAME)
    agent = Agent(tau=TAU, gamma=GAMMA, batch_size=BATCH_SIZE, lr=LEARNING_RATE, state_size=NUM_FRAME, device=DEVICE,
                  double=DOUBLE, prioritized=PRIORITIZED, visual=True, actions_size=env.action_space.n, kappa=KAPPA, N=N)
    reward_log, _ = train(env, agent, num_episode=VISUAL_NUM_EPISODE, eps_init=EPS_INIT, eps_decay=EPS_DECAY,
                          eps_min=EPS_MIN, max_t=MAX_T, num_frame=NUM_FRAME)
    np.save("{}_reward_log".format(VISUAL_ENV_NAME))
    agent.Q_local.to("CPU")
    torch.save(agent.Q_local.state_dict(), "{}_weights.pth".format(VISUAL_ENV_NAME))
