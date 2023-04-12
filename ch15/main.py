import sys
sys.path.append('..')

from ch12.algo import PPO

import gym
import torch
import random
import matplotlib.pyplot as plt
from algo import *
from tqdm import tqdm


def test_agent(agent, env, n_episode):
    return_list = []
    for episode in range(n_episode):
        episode_return = 0
        state = env.reset()
        done = False
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_return += reward
        return_list.append(episode_return)
    return np.mean(return_list)


hidden_dim = 128
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

env_name = 'CartPole-v0'
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = PPO(state_dim, hidden_dim, action_dim, 0, 0, 0, 0, 0, 0, device)
state_dict = torch.load('../ch12/actor_cartpolev0.pth')
agent.actor.load_state_dict(state_dict)
state_dict = torch.load('../ch12/critic_cartpolev0.pth')
agent.critic.load_state_dict(state_dict)

env.seed(0)
torch.manual_seed(0)
random.seed(0)
n_episode = 1
expert_s, expert_a = sample_expert_data(env, agent, n_episode)

n_samples = 30
random_index = random.sample(range(expert_s.shape[0]), n_samples)
expert_s = expert_s[random_index]
expert_a = expert_a[random_index]

env.seed(0)
torch.manual_seed(0)
random.seed(0)

lr = 1e-3
bc_agent = BehaviorClone(state_dim, hidden_dim, action_dim, lr, device)
n_iterations = 1000
batch_size = 64
test_returns = []

# with tqdm(total=n_iterations, desc="Complete") as pbar:
#     for i in range(n_iterations):
#         sample_indices = np.random.randint(low=0, high=expert_s.shape[0], size=batch_size)
#         bc_agent.learn(expert_s[sample_indices], expert_a[sample_indices])
#         current_return = test_agent(bc_agent, env, 5)
#         test_returns.append(current_return)
#         if (i + 1) % 10 == 0:
#             pbar.set_postfix({'return': '%.3f' % np.mean(test_returns[-10:])})
#         pbar.update(1)

# torch.save(bc_agent.policy.state_dict(), 'bc_actor_cartpolev0.pth')
# iteration_list = list(range(len(test_returns)))
# plt.plot(iteration_list, test_returns)
# plt.xlabel('Iterations')
# plt.ylabel('Returns')
# plt.title('BC on {}'.format(env_name))
# plt.show()


actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 250
hidden_dim = 128
gamma = 0.98
lmbda = 0.95
epochs = 10
eps = 0.2
lr_d = 1e-3

env.seed(0)
torch.manual_seed(0)
agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs,
            eps, gamma, device)
gail = GAIL(agent, state_dim, hidden_dim, action_dim, lr_d, device)
n_episode = 500
return_list = []

with tqdm(total=n_episode, desc='Complete') as pbar:
    for i in range(n_episode):
        episode_return = 0
        state = env.reset()
        done = False
        state_list = []
        action_list = []
        next_state_list = []
        done_list = []
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, _ = env.step(action)
            state_list.append(state)
            action_list.append(action)
            next_state_list.append(next_state)
            done_list.append(done)
            state = next_state
            episode_return += reward
        return_list.append(episode_return)
        gail.learn(expert_s, expert_a, state_list, action_list, next_state_list, done_list)
        if (i + 1) % 10 == 0:
            pbar.set_postfix({'return': '%.3f' % np.mean(return_list[-10:])})
        pbar.update(1)

torch.save(gail.agent.actor.state_dict(), 'gail_actor_cartpolev0.pth')
iteration_list = list(range(len(return_list)))
plt.plot(iteration_list, return_list)
plt.xlabel('Iterations')
plt.ylabel('Returns')
plt.title('GAIL on {}'.format(env_name))
plt.show()
