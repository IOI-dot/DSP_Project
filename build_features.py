import numpy as np
import scipy.io as sio
from time_domain import time_domain
from freq_domain import freq_domain

# load subject and choose window length here
data = sio.loadmat('D:\AUC\Spring 2026\DSP\Project\Data\s01.mat') #make sure to change this path to the correct one on your pc @omar
n = 2.0

time_X, valence, arousal = time_domain(data, n=n)
freq_X, _, _ = freq_domain(data, n=n)
combined_X = np.concatenate([time_X, freq_X], axis=1)

print("time-only:    ", time_X.shape)
print("freq-only:    ", freq_X.shape)
print("combined:     ", combined_X.shape)
print("valence:      ", valence.shape)
print("arousal:      ", arousal.shape)