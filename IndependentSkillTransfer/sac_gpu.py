import math
import random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from IPython.display import clear_output
import matplotlib.pyplot as plt
import time
import os
# import pysnooper

#---------Jinxin
import os
import os.path as osp
import joblib

# from utils.mpi_tools import mpi_fork, proc_id, mpi_statistics_scalar, num_procs
# from utils.mpi_torch import average_gradients, sync_all_params
# import argparse
from utils.logx import EpochLogger
from utils.run_utils import setup_logger_kwargs

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, *batch):
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        self.buffer.append(batch)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # state, action, reward, nxt_state, done = map(np.stack, zip(*batch))
        return zip(*batch)

    def __len__(self):
        return len(self.buffer)

# data type: torch.tensor
class Value_Network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state):
        y1 = F.relu(self.linear1(state))
        y2 = F.relu(self.linear2(y1))
        y3 = self.linear3(y2)
        return y3

class Soft_Q_Network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        y1 = F.relu(self.linear1(x))
        y2 = F.relu(self.linear2(y1))
        y3 = self.linear3(y2)
        return y3

class Policy_Network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, LOG_STD_MIN=-20, LOG_STD_MAX=0):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.mean_linear = nn.Linear(hidden_size, output_size)
        self.log_std_linear = nn.Linear(hidden_size, output_size)
        self.log_std_min = LOG_STD_MIN
        self.log_std_max = LOG_STD_MAX

    def forward(self, state):
        # print(state)
        y1 = F.relu(self.linear1(state))
        y2 = F.relu(self.linear2(y1))
        mean = self.mean_linear(y2)
        log_std = torch.clamp(self.log_std_linear(y2), self.log_std_min, self.log_std_max)
        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        unit_normal_distribution = Normal(0, 1)
        unit_sample = unit_normal_distribution.sample()
        normal_sample = mean + unit_sample * log_std.exp()
        action = torch.tanh(normal_sample)
        log_prob = Normal(mean, log_std.exp()).log_prob(normal_sample) \
                   - torch.log(1 - action.pow(2) + epsilon)
        return action, log_prob, normal_sample


class Agent(object):
    def __init__(self, **kwargs):

        for key, value in kwargs.items():
            setattr(self, key, value)

        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        # print(state_dim, action_dim)
        # initialize value funciton
        self.value_Network = Value_Network(state_dim, 256, 1).to(self.device)
        self.value_net_optimizer = optim.Adam(self.value_Network.parameters(), lr=self.value_net_lr)
        self.value_Network_target = Value_Network(state_dim, 256, 1).to(self.device)
        self.value_Network_target.load_state_dict(self.value_Network.state_dict())
        self.value_net_loss_func = nn.MSELoss()
        self.value_net_optimizer.zero_grad()
        # initialize Q funciton
        self.soft_Q_Network = Soft_Q_Network(state_dim + action_dim, 256, action_dim).to(self.device)
        self.soft_Q_net_optimizer = optim.Adam(self.soft_Q_Network.parameters(), lr=self.soft_Q_net_lr)
        self.soft_Q_Network_target = Soft_Q_Network(state_dim + action_dim, 256, action_dim).to(self.device)
        self.soft_Q_Network_target.load_state_dict(self.soft_Q_Network.state_dict())
        self.soft_Q_net_loss_func = nn.MSELoss()
        self.soft_Q_net_optimizer.zero_grad()
        # initialize policy network
        self.policy_Network = Policy_Network(state_dim, 256, action_dim).to(self.device)
        self.policy_net_optimizer = optim.Adam(self.policy_Network.parameters(), lr=self.policy_net_lr)
        self.policy_net_optimizer.zero_grad()
        # initialize replay buffer
        self.replay_Buffer = ReplayBuffer(self.replay_buffer_size)
        # synchronize the parameters of networks in all threads

    def act(self, state):
        # print(state)
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        mean, log_std = self.policy_Network(state_tensor)
        normal_distribution = Normal(mean, log_std.exp())
        action_sample = normal_distribution.sample()
        # action_normalized = torch.softmax(action_sample, dim=0)
        action = torch.tanh(action_sample).squeeze(0).detach().cpu().numpy()
        return action

    def value_Network_backward(self, state, log_prob, soft_Q_value):
        value_predict = self.value_Network(state)
        value_label = soft_Q_value - log_prob # self.temperature *
        value_loss = self.value_net_loss_func(value_predict, value_label.detach())
        value_loss.backward()
        # if self.learn_times % self.ave_gradient_times == self.ave_gradient_times - 1:
        #     average_gradients(self.value_net_optimizer.param_groups) # average the gradients of all threads


    def soft_Q_Network_backward(self, state, action, reward, nxt_state):
        soft_Q_predict = self.soft_Q_Network(state, action)
        soft_Q_label = reward + self.discount * self.value_Network_target(nxt_state)
        soft_Q_loss = self.soft_Q_net_loss_func(soft_Q_predict, soft_Q_label)
        # print(soft_Q_loss)
        soft_Q_loss.backward()
        # if self.learn_times % self.ave_gradient_times == self.ave_gradient_times - 1:
        #     average_gradients(self.soft_Q_net_optimizer.param_groups) # average the gradients of all threads


    def policy_Network_backward(self, log_prob, soft_Q_value):
        policy_loss = -torch.mean(soft_Q_value - self.temperature * log_prob) ## -mean(Q - H) = -mean(V)
        policy_loss.backward()
        # if self.learn_times % self.ave_gradient_times == self.ave_gradient_times - 1:
        #     average_gradients(self.policy_net_optimizer.param_groups) # average the gradients of all threads
            # print(self.learn_times, 'ave gradients')


    def target_network_update(self, target_net, eval_net):
        for target_params, eval_params in zip(target_net.parameters(), eval_net.parameters()):
            target_params.data.copy_(target_params.data * (1.0 - self.target_update) + eval_params * self.target_update)

    def backward(self):
        if self.replay_Buffer.__len__() < self.batch_size:
            return

        state, action, reward, nxt_state = self.replay_Buffer.sample(self.batch_size)
        state_tensor = torch.tensor(state, dtype=torch.float).to(self.device)
        action_tensor = torch.tensor(action, dtype=torch.float).to(self.device)
        reward_tensor = torch.tensor(reward, dtype=torch.float).unsqueeze(1).to(self.device)
        nxt_state_tensor = torch.tensor(nxt_state, dtype=torch.float).to(self.device)

        action_sample, log_prob, _ = self.policy_Network.evaluate(state_tensor)
        # soft_Q_value = self.soft_Q_Network(state_tensor, action_sample)
        soft_Q_value = self.soft_Q_Network_target(state_tensor, action_sample)

        self.value_Network_backward(state_tensor, log_prob, soft_Q_value)
        self.soft_Q_Network_backward(state_tensor, action_tensor, reward_tensor, nxt_state_tensor)
        self.policy_Network_backward(log_prob, soft_Q_value)

        self.target_network_update(self.value_Network_target, self.value_Network)
        self.target_network_update(self.soft_Q_Network_target, self.soft_Q_Network)

    def step(self):
        self.value_net_optimizer.step()
        self.soft_Q_net_optimizer.step()
        self.policy_net_optimizer.step()
        self.value_net_optimizer.zero_grad()
        self.soft_Q_net_optimizer.zero_grad()
        self.policy_net_optimizer.zero_grad()

    def reserve_network(self, folder, episode):
        torch.save(self.value_Network.state_dict(), folder + 'E' + str(episode) + '_sac_value_Network.pkl')  #
        torch.save(self.soft_Q_Network.state_dict(), folder + 'E' + str(episode) + '_soft_Q_Network.pkl')  #
        torch.save(self.policy_Network.state_dict(), folder + 'E' + str(episode) + '_policy_Network.pkl')  #
        # np.savetxt(folder + 'reward.txt', self.reward_buf)

    def load_network(self, folder, episode):
        self.value_Network.load_state_dict(torch.load(folder + 'E' + str(episode) + '_sac_value_Network.pkl'))
        self.soft_Q_Network.load_state_dict(torch.load(folder + 'E' + str(episode) + '_soft_Q_Network.pkl'))
        self.policy_Network.load_state_dict(torch.load(folder + 'E' + str(episode) + '_policy_Network.pkl'))
        self.value_Network_target.load_state_dict(self.value_Network.state_dict())
        self.soft_Q_Network_target.load_state_dict(self.soft_Q_Network.state_dict())
        # for target_params, eval_params in zip(self.soft_Q_Network_target.parameters(), self.soft_Q_Network.parameters()):
        #     target_params.data.copy_(eval_params)
        # for target_params, eval_params in zip(self.value_Network_target.parameters(), self.value_Network.parameters()):
        #     target_params.data.copy_(eval_params)
        # self.reward_buf = list(np.loadtxt(folder + 'reward.txt'))

    # def sync_multi_thread(self):
    #     # synchronize the parameters of networks in all threads
    #     sync_all_params(self.value_Network.parameters())
    #     sync_all_params(self.soft_Q_Network.parameters())
    #     sync_all_params(self.policy_Network.parameters())
    #     sync_all_params(self.value_Network_target.parameters())
    #     sync_all_params(self.soft_Q_Network_target.parameters())

    def logger_setup(self, logger_kwargs, **kwargs):
        self.logger = EpochLogger(**logger_kwargs)
        for key, value in kwargs.items():
            if key != 'env' and key != 'output_dir':
                self.logger.log_tabular(key, value)

    def logger_update(self, kwargs):
        for key, value in kwargs.items():
            self.logger.log_tabular(key, value)
        # print(self.logger.log_headers)
        self.logger.dump_tabular()

class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action

params = {
        'discount': 0.99,
        'value_net_lr': 1e-3,
        'soft_Q_net_lr': 1e-3,
        'policy_net_lr': 1e-3,
        'replay_buffer_size': 1e5,
        'batch_size': 256,
        'target_update': 2e-2,
        'temperature': 0.5,
    }

if __name__ == "__main__":

    env_name = 'hopper_v3_hurdle'
    output_dir = './network_reserve/sac_gpu/'

    # gpu_init
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = NormalizedActions(gym.make(env_name))
    params.update({
        'env': env,
        'output_dir': output_dir,
        'device': device,
    })

    logger_kwargs = setup_logger_kwargs(exp_name=env_name, seed=0, data_dir=output_dir)
    agent = Agent(**params)
    agent.logger_setup(logger_kwargs, **params)

    MAX_EPISODE = 100 # 100 # 400
    MAX_EPOCH = 10
    MAX_STEP = 500 # 1000 # 2000 #

    for episode_idx in range(MAX_EPISODE):
        episode_reward = 0
        episode_success = 0
        for epoch in range(MAX_EPOCH):
            state = env.reset()
            for step in range(MAX_STEP):
                # env.render()
                action = agent.act(state)
                nxt_state, reward, done, info = env.step(action)
                agent.replay_Buffer.push(state, action, reward, nxt_state)
                agent.backward()
                agent.step()
                state = nxt_state
                episode_reward += reward
                episode_success += info['success']
                if done:
                    # print(epoch, step, done)
                    break
                # print(step)
        agent.reserve_network(output_dir + env_name + '/', episode_idx)
        agent.logger_update({
            'episode': episode_idx,
            'avg_reward': episode_reward,
            'success': episode_success,
        })
        # print('episode_idx: ', episode_idx, 'reward: ', episode_reward)

    # Agent.reserve_network(output_dir)












