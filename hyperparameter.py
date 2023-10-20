import torch

print_step = 10
render_step = 200
n_training_episodes = 1000
gamma = 0.99
lr_a = 0.0005
lr_c = 0.0001
env_id = 'LunarLander-v2'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_save_name = 'LunarLander'
frame_render = 5