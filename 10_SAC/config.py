import torch

#device, CPU or the GPU of your choice
GPU = 0
DEVICE = torch.device("cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu")

#environment names
RAM_ENV_NAME = 'LunarLanderContinuous-v2'

#Agent parameters
N = 5
LEARNING_RATE = 0.001
ALPHA = 0.01
SCALE = 1
STD = 3
UPDATE_FREQUENCY = 2
TAU = 0.005
BATCH_SIZE = 256
LOG_IN_V = True
LOG_IN_PI = True
BUFFER_SIZE = int(1e6)

#Training parameters
RAM_NUM_EPISODE = 1000
MAX_T = 1500