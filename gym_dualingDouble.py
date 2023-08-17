import gym
import numpy as np

from DeepQ_duelingDouble import Agent

from utils import plot_learning_curve


if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    num_games = 500
    load_checkpoint = False

    agent = Agent(gamma=0.99, epsilon=1.0, lr=5e-4, input_dims=[8], n_actions=4, mem_size=10000000, 
                    eps_min=0.01, batch_size=64, eps_dec=1e-3, replace=100)
    
    if load_checkpoint:
        agent.load_models()

    filename = 'LunarLander-DuelingDDQN-Adam-lr0005-replace100.png'

    scores, eps_history = [], []

    for i in range(num_games):
        done = False
        observation = env.reset()
        score = 0

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, int(done))

            agent.learn()
            observation = observation_

        scores.append(score)
        avg_score = np.mean(scores[-100:])

        print('episode ', i, 'score %.1f' % score,
                'average score %.1f' % avg_score, 
                'epsilon %.3f' % agent.epsilon)

        if i>10 and i % 10 == 0:
            agent.save_models()

        eps_history.append(agent.epsilon)

    x = [i+1 for i in range(num_games)]
    plot_learning_curve(x, scores, eps_history, filename)



