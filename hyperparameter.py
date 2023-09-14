import torch

print_step = 100
render_step = 250
n_training_episodes = 1000
max_t = 1000
max_t_sim = 100
gamma = 0.99
lr_a = 0.0001
lr_c = 0.001
env_id = 'HalfCheetah-v4'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
t_name = 'train'
e_name = 'eval'
model_name = 'HalfCheetah'