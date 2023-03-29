import gym

from algo import PolicyIteration, ValueIteration
from env import CliffWalkingEnv
from utils import print_agent


env = CliffWalkingEnv()
action_meaning = ['^', 'v', '<', '>']
theta = 0.001
gamma = 0.9

agent = PolicyIteration(env, theta, gamma)
agent.policy_iteration()
print_agent(agent, action_meaning, list(range(37, 47)), [47])

print('-' * 80)
agent = ValueIteration(env, theta, gamma)
agent.value_iteration()
print_agent(agent, action_meaning, list(range(37, 47)), [47])

print('=' * 80)
env = gym.make('FrozenLake-v1')
env.reset()
action_meaning = ['<', 'V', '>', '^']
theta = 1e-5
gamma = 0.9

agent = PolicyIteration(env, theta, gamma)
agent.policy_iteration()
print_agent(agent, action_meaning, [5, 7, 11, 12], [15])

print('-' * 80)
env.reset()
agent = ValueIteration(env, theta, gamma)
agent.value_iteration()
print_agent(agent, action_meaning, [5, 7, 11, 12], [15])