import torch

# Device, CPU or the GPU of your choice
GPU = 0
DEVICE = torch.device("cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu")

# environment names
RAM_ENV_NAME = 'LunarLanderContinuous-v2'
VISUAL_ENV_NAME = None

# Agent parameters
BATCH_SIZE = 100
LR = 0.001
STEP = 1
DELAY = 2
TAU = 0.005
EXPLORATION_LEVEL = 0.2
SMOOTHING_LEVEL = 0.2
SMOOTHING_MAX = 0.5
BN = False,
CLIP = True,
INIT = True,
HIDDEN = [400, 300]

# Training parameters
L = 1000
N = 10