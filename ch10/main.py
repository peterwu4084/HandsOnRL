import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from algo import *
from rl_utils import *
from tqdm import tqdm


actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 1000
hidden_dim = 128
gamma = 0.98
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

env_name = 'CartPole-v0'
env = gym.make(env_name)
env.seed(0)
torch.manual_seed(0)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device)

return_list = train_on_policy_agent(env, agent, num_episodes)
torch.save(agent.actor.state_dict(), 'actor_cartpolev0.pth')
torch.save(agent.critic.state_dict(), 'critic_cartpolev0.pth')

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('ActorCritic on {}'.format(env_name))
plt.show()

mv_return = moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('ActorCritic on {}'.format(env_name))
plt.show()
