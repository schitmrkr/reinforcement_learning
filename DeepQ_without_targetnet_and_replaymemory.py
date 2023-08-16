import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

class DeepQNet(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNet, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.lc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.lc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.lc3 = nn.Linear(self.fc2_dims, self.n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    def forward(self, state):
        x = F.relu(self.lc1(state))
        x = F.relu(self.lc2(x))
        actions = self.fc3(x)
        return actions



class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size=1000, eps_end = 0.01, eps_dec = 5e-4):   
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]

        self.mem_size = max_mem_size
        self.mem_counter = 0

        self.batch_size = batch_size
        self.input_dims = input_dims

        self.Q_evals = DeepQNet(self.lr, n_actions=n_actions, input_dims=input_dims, fc1_dims=256, fc2_dims=256)

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype = np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype = np.float32)

        self.action_memory = np.zeros((self.mem_size), dtype = np.int32)
        self.reward_memory = np.zeros((self.mem_size), dtype=np.float32)

        self.terminal_memory =  np.zeros((self.mem_size), dtype=np.bool)


    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_counter += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor([observation]).to(self.Q_evals.device)
            actions = self.Q_evals.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.mem_counter < self.batch_size:
            return
        
        self.Q_evals.optimizer.zero_grad()

        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_evals.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.Q_evals.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.Q_evals.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.Q_evals.device)

        action_batch = self.action_memory[batch]

        q_eval = self.Q_evals.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_evals.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * torch.max(q_next, axis=1)[0]

        loss = self.Q_evals.loss(q_target, q_eval).to(self.Q_evals.device)
        loss.backward()
        self.Q_evals.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
                        else self.eps_min

        
















