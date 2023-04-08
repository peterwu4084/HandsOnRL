import gym
import torch
from algo import *
from time import sleep


hidden_dim = 128
gamma = 0.98
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

env = gym.make('CartPole-v0')

agent = TRPO(hidden_dim, env.observation_space, env.action_space, 0,
             0, 0, 0, gamma, device)
state_dict = torch.load('actor_cartpolev0.pth')
agent.actor.load_state_dict(state_dict)
state_dict = torch.load('critic_cartpolev0.pth')
agent.critic.load_state_dict(state_dict)

state = env.reset()
done = False
agent_return = 0
while not done:
    action = agent.take_max_action(state)
    next_state, reward, done, _ = env.step(action)
    agent_return += reward
    env.render()
    state = next_state
    sleep(0.01)

print('Agent return:', agent_return)


env = gym.make('Pendulum-v1')

agent = TRPOContinuous(hidden_dim, env.observation_space, env.action_space, 0,
             0, 0, 0, gamma, device)
state_dict = torch.load('actor_pendulumv1.pth')
agent.actor.load_state_dict(state_dict)
state_dict = torch.load('critic_pendulumv1.pth')
agent.critic.load_state_dict(state_dict)

state = env.reset()
done = False
agent_return = 0
while not done:
    action = agent.take_max_action(state)
    next_state, reward, done, _ = env.step(action)
    agent_return += reward
    env.render()
    state = next_state
    sleep(0.01)

print('Agent return:', agent_return)