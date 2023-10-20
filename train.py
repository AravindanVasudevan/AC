import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from A2C import A2C_Agent
from hyperparameter import(
    print_step,
    render_step,
    n_training_episodes,
    gamma,
    lr_a,
    lr_c,
    env_id
)

if __name__ == '__main__':
    env = gym.make(env_id)
    eval_env = gym.make(env_id, render_mode = 'rgb_array')

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = A2C_Agent(lr_a, lr_c, gamma, state_size, action_size)

    total_reward_list = []

    for eps in range(1, n_training_episodes + 1):
        state, _ = env.reset()
        total_reward = 0

        while True:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            action = torch.tensor(action)
            agent.train(state, next_state, action, reward)

            done = terminated or truncated

            if total_reward <= -250:
                done = True

            if done:
                break

            state = next_state
        
        total_reward_list.append(total_reward)
        
        if eps % print_step == 0:
            print(f'Reward obtained at episode {eps} is {total_reward}')
        
        if eps % render_step == 0:
            print(f'Saving simulation...')
            agent.render(eps, eval_env)
            print(f'Simulation saved!')
    
    episodes = np.arange(1, n_training_episodes + 1)

    plt.plot(episodes, total_reward_list, label = 'Total Reward per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.legend()
    plt.title('Training Performance')
    plt.show()