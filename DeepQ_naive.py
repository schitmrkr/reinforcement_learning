import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from utils import plot_learning_curve


class LinearDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims):
        super(LinearDeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        layer1 = F.relu(self.fc1(state))
        actions = F.relu(self.fc2(layer1))

        return actions


class Agent():
    def __init__(self, input_dims, n_actions, lr, 
                    gamma=0.99, epsilon=1.0, 
                    eps_dec=1e-5, eps_min=0.01):
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min

        self.action_space = [i for i in range(self.n_actions)]

        self.Q = LinearDeepQNetwork(self.lr, self.n_actions, self.input_dims)

    
    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor(observation, dtype=torch.float).to(self.Q.device)
            actions = self.Q.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action
    
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                        if self.epsilon > self.eps_min else self.eps_min


    def learn(self, state, action, reward, state_):
        self.Q.optimizer.zero_grad()
        states = torch.tensor(state, dtype=torch.float).to(self.Q.device)
        actions = torch.tensor(action).to(self.Q.device)
        rewards = torch.tensor(reward).to(self.Q.device)
        states_ = torch.tensor(state_, dtype=torch.float).to(self.Q.device)
        
        q_pred = self.Q.forward(states)[actions]

        q_next = self.Q.forward(states_).max()

        q_target = reward + self.gamma * q_next

        loss = self.Q.loss(q_target, q_pred).to(self.Q.device)
        loss.backward()
        self.Q.optimizer.step()
        self.decrement_epsilon()


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    n_games = 10000
    scores = []
    eps_history = []

    agent = Agent(lr=0.0001, input_dims=env.observation_space.shape, n_actions=env.action_space.n)

    for i in range(n_games):
        score = 0
        done = False
        obs = env.reset()
        obs = obs[0]

        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, truncated, info = env.step(action)
            
            score += reward
            agent.learn(obs, action, reward, obs_)
            obs = obs_
        
        scores.append(score)
        eps_history.append(agent.epsilon)

        if i % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print('episode ', i, 'score %.1f avg score %.1f epsilon %.2f ' % (score, avg_score, agent.epsilon))

    filename = 'cartpole_naive_dqn.png'
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, scores, eps_history, filename)     









        
