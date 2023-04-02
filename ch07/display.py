import gym
import torch
from algo import DQN
from time import sleep


lr = 2e-3
hidden_dim = 128
gamma = 0.98
epsilon = 0.0
target_update = 10
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

env = gym.make('CartPole-v0')

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)
state_dict = torch.load('dqn_cartpolev0.pth')
agent.q_net.load_state_dict(state_dict)
agent.target_q_net.load_state_dict(state_dict)

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