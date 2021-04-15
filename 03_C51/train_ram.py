import gym
from config import *
from agent import *
from utils import *


def train(env, agent, num_episode, eps_init, eps_decay, eps_min, max_t):
    rewards_log = []
    average_log = []
    eps = eps_init

    for i in range(1, 1 + num_episode):
        env = gym.make(RAM_ENV_NAME)
        state = env.reset()
        episode_reward = 0
        done = False
        t = 0

        while not done and t < max_t:
            t += 1
            state = state.reshape(1, -1)
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            transition = (state, action, reward, next_state, done)
            agent.memory.remember(transition)

            if len(agent.memory) >= agent.batch_size:
                agent.learn()

            if t % 4 == 0 and len(agent.memory) >= agent.batch_size:
                agent.soft_update()
            state = next_state.copy()
            episode_reward += reward

        rewards_log.append(episode_reward)
        average_log.append(np.mean(rewards_log[-100:]))
        print('\rEpisode {}, Reward {:.3f}, Average Reward {:.3f}'.format(i, episode_reward, average_log[-1]), end='')
        if i % 100 == 0:
            print()
        eps = max(eps_decay * eps, eps_min)
    return rewards_log, average_log


if __name__ == "__main__":
    env = gym.make(RAM_ENV_NAME)
    agent = Agent(tau=TAU, gamma=GAMMA, batch_size=BATCH_SIZE, lr=LEARNING_RATE,
                  state_size=env.observation_space.shape[0],
                  actions_size=env.action_space.n, device=DEVICE, double=DOUBLE, visual=False,
                  prioritized=PRIORITIZED, v_min=V_MIN, v_max=V_MAX, N=N)
    rewards_log, _ = train(env, agent, num_episode=RAM_NUM_EPISODE, eps_init=EPS_INIT, eps_decay=EPS_DECAY,
                           eps_min=EPS_MIN, max_t=MAX_T)
    np.save("{}_rewards_log".format(RAM_ENV_NAME), rewards_log)
    agent.Q_local.to('cpu')
    torch.save(agent.Q_local.state_dict(), "{}_weights.pth".format(RAM_ENV_NAME))
