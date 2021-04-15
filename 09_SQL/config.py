import torch

# device, CPU or the GPU of your choice
GPU = 0
DEVICE = torch.device("cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu")

# environment names
RAM_ENV_NAME = 'LunarLander-v2'
VISUAL_ENV_NAME = 'Pong-v0'
CONSTANT = 90

# Agent parameters
ALPHA = 0.05
BATCH_SIZE = 128
LEARNING_RATE = 0.001
TAU = 0.001
GAMMA = 0.99

# Training parameters
RAM_NUM_EPISODE = 1000
VISUAL_NUM_EPISODE = 3000
MAX_T = 1500
NUM_FRAME = 2