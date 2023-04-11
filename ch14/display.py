import gym
import torch
from algo import *
from time import sleep


hidden_dim = 128
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]
agent = SACContinuous(state_dim, hidden_dim, action_dim, action_bound, 0,
                      0, 0, 0, 0, 0, device)
state_dict = torch.load('actor_pendulumv1.pth')
agent.actor.load_state_dict(state_dict)

state = env.reset()
done = False
agent_return = 0
while not done:
    action = agent.take_action(state)[0]
    next_state, reward, done, _ = env.step([action])
    agent_return += reward
    env.render()
    state = next_state
    sleep(0.01)

print('Agent return:', agent_return)


env = gym.make('CartPole-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = SAC(state_dim, hidden_dim, action_dim, 0, 0, 0, 0, 0, 0, device)
state_dict = torch.load('actor_cartpolev0.pth')
agent.actor.load_state_dict(state_dict)

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