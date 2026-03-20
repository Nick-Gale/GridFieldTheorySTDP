## Training loop for STDP based cortical network
from src.neural import *
import numpy as np
import cupy as cp
import tqdm 

use_gpu = True
xp = np if not use_gpu else cp

data = xp.array(np.load("data/data.npy"))
nepochs = 1
integration_time = 30 # number of activity timesteps to integrate into the learning rule
batches = xp.shape(data)[2] // integration_time # batches are serialised, not shuffled, training is serial not parallel (these can be examined in future)

cortical_system = CorticalSystem(shape=(data.shape[0],data.shape[1]), dt=0.01, T=integration_time)
print(type(data))
for n in range(nepochs):
    for S in tqdm.tqdm(range(batches)):
        # compute activity
        for t in tqdm.tqdm(range(integration_time)):
            current = data[::, ::, S * integration_time + t]
            print(type(current))
            cortical_system.propagate(current, integration_time)

        # learning step
        cortical_system.learn()


# save the data
weights = cortical_system._W_flat
np.save("models/model_weight.npy", weights)
