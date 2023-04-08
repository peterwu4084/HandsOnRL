import gym
import torch
import matplotlib.pyplot as plt
from algo import *

import sys
sys.path.append('..')
from rl_utils import *


num_episodes = 1000
hidden_dim = 128
gamma = 0.98
lmbda = 0.95
critic_lr = 1e-2
kl_constraint = 0.0005
alpha = 0.5
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# env_name = 'CartPole-v0'
# env = gym.make(env_name)
# env.seed(0)
# torch.manual_seed(0)

# agent = TRPO(hidden_dim, env.observation_space, env.action_space, lmbda,
#              kl_constraint, alpha, critic_lr, gamma, device)
# return_list = train_on_policy_agent(env, agent, num_episodes)

# torch.save(agent.actor.state_dict(), 'actor_cartpolev0.pth')
# torch.save(agent.critic.state_dict(), 'critic_cartpolev0.pth')

# episodes_list = list(range(len(return_list)))
# plt.plot(episodes_list, return_list)
# plt.xlabel('Episodes')
# plt.ylabel('Returns')
# plt.title('TPRO on {}'.format(env_name))
# plt.show()

# mv_return = moving_average(return_list, 9)
# plt.plot(episodes_list, mv_return)
# plt.xlabel('Episodes')
# plt.ylabel('Returns')
# plt.title('TPRO on {}'.format(env_name))
# plt.show()


num_episodes = 2000
hidden_dim = 128
gamma = 0.9
lmbda = 0.9
critic_lr = 1e-2
kl_constraint = 0.0002
alpha = 0.5

env_name = 'Pendulum-v1'
env = gym.make(env_name)
env.seed(0)
torch.manual_seed(0)
agent = TRPOContinuous(hidden_dim, env.observation_space, env.action_space, lmbda, kl_constraint,
                       alpha, critic_lr, gamma, device)
return_list = train_on_policy_agent(env, agent, num_episodes)
torch.save(agent.actor.state_dict(), 'actor_pendulumv1.pth')
torch.save(agent.critic.state_dict(), 'critic_pendulumv1.pth')

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('TPRO on {}'.format(env_name))
plt.show()

mv_return = moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('TPRO on {}'.format(env_name))
plt.show()