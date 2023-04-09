import gym
import torch
from algo import *
from time import sleep


hidden_dim = 64
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]
agent = DDPG(state_dim, action_dim, state_dim + action_dim, hidden_dim, False,
             action_bound, 0, 0, 0, 0, 0, device)
state_dict = torch.load('actor_pendulumv1.pth')
agent.actor.load_state_dict(state_dict)
state_dict = torch.load('critic_pendulumv1.pth')
agent.critic.load_state_dict(state_dict)

state = env.reset()
done = False
agent_return = 0
while not done:
    action = agent.take_action(state)
    next_state, reward, done, _ = env.step(action)
    agent_return += reward
    env.render()
    state = next_state
    sleep(0.01)

print('Agent return:', agent_return)