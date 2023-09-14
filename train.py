import gym
import imageio
import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions import Normal
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import (
    Actor,
    Critic
)
from hyperparameter import(
    print_step,
    render_step,
    n_training_episodes,
    max_t,
    max_t_sim,
    gamma,
    lr_a,
    lr_c,
    env_id,
    device,
    t_name,
    e_name,
    model_name
)

def train(e, env, name, max_t, max_t_sim, actor, critic, opt_actor, opt_critic, print_step, render_step, gamma):
    state, _ = env.reset()
    total_reward = 0

    actor.train()
    critic.train()

    for _ in range(max_t):
        mean, stddev = actor(state)
        stddev = stddev + 1e-8
        action = torch.normal(mean, stddev)
        action_dist = Normal(mean, stddev)
        log_prob = action_dist.log_prob(action)
        log_prob = log_prob.sum()
        action = action.squeeze().detach().cpu().numpy()

        next_state, reward, terminated, _, _ = env.step(action)

        if terminated:
            break

        value_curr_state = critic(state)
        value_next_state = critic(next_state)

        advantage = reward + ((1 - terminated) * gamma * value_next_state) - value_curr_state                                    
                                             
        total_reward += reward
        state = next_state
       
        critic_loss = 0.5 * (value_curr_state - (reward + (gamma * value_next_state))).pow(2).mean()
        opt_critic.zero_grad()
        critic_loss.backward()
        opt_critic.step()
        
        actor_loss = -log_prob * advantage.detach() # actor loss = negative of the log probability of the action taken, multiplied by the advantage
        opt_actor.zero_grad()
        actor_loss.backward()
        opt_actor.step()
        
    if e % render_step == 0 or e == 1:
        print(f'Episode {e} Reward: {total_reward}')

        actor.eval()
        frames = []
        state, _ = env.reset()
        for _ in range(max_t_sim):
            mean, stddev = actor(state)
            action = torch.normal(mean, stddev)
            action = action.squeeze().detach().cpu().numpy()

            next_state, reward, terminated, _, _ = env.step(action)  
            frame = env.render()
            frames.append(frame)
            imageio.mimsave(f'simulations/{name}_simulation_episode_{e}.gif', frames)

            if terminated:
                break
            state = next_state

        print(f'simulation for training episode {e} is saved')

    return total_reward

if __name__ == '__main__':
    env = gym.make(env_id, render_mode = 'rgb_array')
    eval_env = gym.make(env_id, render_mode = 'rgb_array')
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    actor = Actor(state_size, action_size).to(device)
    critic = Critic(state_size).to(device)
    opt_actor = torch.optim.Adam(actor.parameters(), lr = lr_a)
    opt_critic = torch.optim.Adam(critic.parameters(), lr = lr_c)

    rewards = []
    best_reward = 0

    for e in range(1, n_training_episodes + 1):
        r = train(e, env, t_name, max_t, max_t_sim, actor, critic, opt_actor, opt_critic, print_step, render_step, gamma)
        rewards.append(r)

        if r > best_reward or e == 1:
            torch.save({'model_state_dict': actor.state_dict()}, f'checkpoints/{model_name}_actor_checkpoint.pth')
            torch.save({'model_state_dict': critic.state_dict()}, f'checkpoints/{model_name}_critic_checkpoint.pth')
            print('Saving the best model')

    print('Done!')