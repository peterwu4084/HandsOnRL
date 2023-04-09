import gym
import torch
import matplotlib.pyplot as plt
import random
import sys
sys.path.append('..')
from rl_utils import *
from algo import DDPG

actor_lr = 5e-4
critic_lr = 5e-3
num_episodes = 200
hidden_dim = 64
gamma = 0.98
tau = 0.005
buffer_size = 10000
minimal_size = 1000
batch_size = 64
sigma = 0.01
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


env_name = 'Pendulum-v1'
env = gym.make(env_name)
random.seed(0)
np.random.seed(0)
env.seed(0)
torch.manual_seed(0)
replay_buffer = ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]
agent = DDPG(state_dim, action_dim, state_dim + action_dim, hidden_dim, False,
             action_bound, sigma, actor_lr, critic_lr, tau, gamma, device)

return_list = train_off_policy_agent(env, agent, num_episodes,
                                     replay_buffer, minimal_size,
                                     batch_size)

torch.save(agent.actor.state_dict(), 'actor_pendulumv1.pth')
torch.save(agent.critic.state_dict(), 'critic_pendulumv1.pth')

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DDPG on {}'.format(env_name))
plt.show()

mv_return = moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DDPG on {}'.format(env_name))
plt.show()
