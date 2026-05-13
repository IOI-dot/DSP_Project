import numpy as np
import scipy.io as sio
def time_domain(EEG: dict, n: float = 0.5) -> tuple:
    fs = EEG["fs"][0][0]
    n_samples = int(n * fs)
    n_trials, n_channels, n_total_samples = EEG["data"].shape

    n_windows = n_total_samples // n_samples
    n_features = 6

    stat_desc = np.zeros((n_trials * n_windows, n_channels * n_features))
    
    arousal = np.repeat([label[1] for label in EEG["labels"]], n_windows)
    valence = np.repeat([label[0] for label in EEG["labels"]], n_windows)

    for i in range(n_trials):
        for j in range(n_channels):
            waves = EEG["data"][i,j]
            offset = 0
            wind_idx = 0
            
            while offset + n_samples <= waves.shape[0]:
                window = waves[offset:offset + n_samples]
                
                # Feature calculations
                mean = window.mean()
                median = np.median(window)
                variance = window.var()
                std = window.std()
                skewness = (np.mean((window - mean)**3)) / (std**3 + 1e-8) # Adding small value to avoid division by zero
                kurtosis = (np.mean((window - mean)**4)) / (std**4 + 1e-8) - 3 
                
                # row: which window are we in 
                row = (i * n_windows) + wind_idx 

                # Column: which group of features (channel) are we in?
                col_start = j * n_features
                col_end = col_start + n_features
                
                stat_desc[row, col_start:col_end] = [mean, median, variance, std, skewness, kurtosis]
                
                offset += n_samples
                wind_idx += 1
                
    return np.array(stat_desc), np.array(valence), np.array(arousal)

if __name__ == "__main__":
    data = sio.loadmat('s01.mat')
    print(data["data"].shape)
    stat_desc, valence, arousal = time_domain(data)
    print(stat_desc.shape)
    print(valence.shape)
    print(arousal.shape)