import torch.optim as optim
from networks import *
from utils import *


class Agent(object):
    def __init__(self, env, lr1=0.0001, lr2=0.001, tau=0.001, gamma=0.99, step=1, speed1=1, speed2=1, learning_time=1,
                 batch_size=64, OUN_noise=True, batch_norm=True, clip=True, initialize=True, hidden=[256, 256],
                 buffer_size=int(1e6)):

        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        self.env = env

        self.lr1 = lr1
        self.lr2 = lr2
        self.tau = tau
        self.gamma = gamma
        self.step = step
        self.speed1 = speed1  # update times per learn
        self.speed2 = speed2
        self.learning_time = learning_time
        self.batch_norm = batch_norm
        self.clip = clip
        self.initialize = initialize
        self.hidden = hidden
        self.memory = ReplayBuffer(max_len=buffer_size)
        self.batch_size = batch_size
        self.OUN_noise = OUN_noise
        self.noise = OUNnoise(action_size=self.action_size)

        self.actor_local = Actor(self.state_size, self.action_size, self.batch_norm, self.initialize).to(DEVICE)
        self.actor_target = Actor(self.state_size, self.action_size, self.batch_norm, self.initialize).to(DEVICE)
        self.critic_local = Critic(self.state_size, self.action_size, self.batch_norm, self.initialize).to(DEVICE)
        self.critic_target = Critic(self.state_size, self.action_size, self.batch_norm, self.initialize).to(DEVICE)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr1)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr2)

        self.soft_update(self.actor_local, self.actor_target, 1)  # keep the same parameters with actor_local
        self.soft_update(self.critic_local, self.critic_target, 1)  # keep the same parameters with critic_local

    def act(self, state, i):
        state = torch.tensor(state, dtype=torch.float).view(1, -1).to(DEVICE)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).detach().view(-1).cpu().numpy()
        if self.OUN_noise:
            noise = self.noise.sample()
        else:
            noise = np.random.standard_normal(size=self.action_size)
        action += noise / math.sqrt(i)
        action = np.clip(action, -1, 1)
        return action

    def soft_update(self, local_model: nn.Module, target_model: nn.Module, tau):
        """
        θ_target = τ*θ_local + (1 - τ)*θ_target
        :param local_model: PyTorch model (weights will be copied from)
        :param target_model:  PyTorch model (weights will be copied to)
        :param tau: interpolation parameter
        """
        for local_layer, target_layer in zip(local_model.modules(), target_model.modules()):
            for local_parameter, target_parameter in zip(local_layer.parameters(), target_layer.parameters()):
                target_parameter.data.copy_(tau * local_parameter.data + (1 - tau) * target_parameter.data)

            try:
                target_layer.running_mean = tau * local_layer.running_mean + (1 - tau) * target_layer.running_mean
                target_layer.running_var = tau * local_layer.running_var + (1 - tau) * target_layer.running_var
            except Exception as e:
                pass

    def learn(self):
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size=self.batch_size)

        with torch.no_grad():
            expected_values = rewards + self.gamma * (1 - dones) * self.critic_target(next_states,
                                                                                      self.actor_target(next_states))

        for _ in range(self.speed1):
            observed_values = self.critic_local(states, actions)
            Loss1 = (expected_values - observed_values).pow(2).mean()
            self.critic_optimizer.zero_grad()  # clean up the deviation in critic
            Loss1.backward()
            if self.clip:
                torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
            self.critic_optimizer.step()

        for _ in range(self.speed2):
            Loss2 = -self.critic_local(states, self.actor_local(states)).mean()
            self.actor_optimizer.zero_grad()
            Loss2.backward()
            if self.clip:
                torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
            self.actor_optimizer.step()

        self.soft_update(self.actor_local, self.actor_target, self.tau)
        self.soft_update(self.critic_local, self.critic_target, self.tau)