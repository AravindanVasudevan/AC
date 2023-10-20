import imageio
import torch
import torch.optim as optim
import numpy as np
from model import Actor, Critic
from hyperparameter import device, model_save_name, frame_render

class A2C_Agent():
    
    def __init__(self, learning_rate_actor, learning_rate_critic, discount, state_size, action_size, hidden_size = 128):
        self.action_size = action_size
        self.actor_network = Actor(state_size, action_size, hidden_size).to(device)
        self.optimizer_actor = optim.Adam(self.actor_network.parameters(), lr = learning_rate_actor)
        self.critic_network = Critic(state_size, hidden_size).to(device)
        self.optimizer_critic = optim.Adam(self.critic_network.parameters(), lr = learning_rate_critic)
        self.discount = discount

    def act(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            action_probs = self.actor_network(state)
            action_probs = action_probs.detach().cpu().squeeze(0).numpy()
            action = np.random.choice(self.action_size, p = action_probs)
            
        return action
    
    def train(self, state, next_state, action, reward):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(device)
        value_curr = self.critic_network(state)
        value_next = self.critic_network(next_state)
        advantage = reward + (self.discount * value_next) - value_curr

        critic_loss = 0.5 * advantage.pow(2).mean()
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        action_probs = self.actor_network(state)
        selected_action_prob = action_probs[0, action]

        actor_loss = -torch.log(selected_action_prob) * advantage.detach()
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()
    
    def save_model(self):
        torch.save({'model_state_dict': self.policy_network.state_dict()}, f'checkpoints/{model_save_name}.pth')
    
    def render(self, eps, eval_env):
        frames = []
        total_reward = 0
        step = 1
        state, _ = eval_env.reset()
        with torch.no_grad():
            while True:
                action = self.act(state)
                next_state, reward, terminated, truncated, _ = eval_env.step(action)
                total_reward += reward
                step += 1
                if step % frame_render == 0:
                    frame = eval_env.render()
                    frames.append(frame)

                done = terminated or truncated

                if total_reward <= -250:
                    done = True

                if done:
                    break

                state = next_state

        imageio.mimsave(f'simulations/{model_save_name}_{eps}.gif', frames)