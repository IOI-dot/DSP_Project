import numpy as np
import scipy.io as sio
def time_domain(EEG: dict, n: float = 0.5) -> tuple:
    fs = EEG["fs"][0][0]
    n_samples = int(n * fs)
    n_trials = len(EEG["data"])
    n_channels = len(EEG["data"][0])
    n_windows = EEG["data"][0][0].shape[0] // n_samples
    n_features = 7

    stat_desc = np.zeros((n_trials * n_windows, n_channels * n_features))
    
    arousal = np.repeat([label[1] for label in EEG["labels"]], n_windows)
    valence = np.repeat([label[0] for label in EEG["labels"]], n_windows)

    for i in range(n_trials):
        for j in range(n_channels):
            waves = EEG["data"][i][j]
            offset = 0
            wind_idx = 0
            
            while offset + n_samples <= waves.shape[0]:
                window = waves[offset:offset + n_samples]
                
                # Feature calculations
                mean = window.mean()
                std = window.std()
                rms = np.sqrt(np.mean(window**2))
                mx = np.max(window)
                skewness = (np.mean((window - mean)**3)) / (std**3)
                kurtosis = (np.mean((window - mean)**4)) / (std**4) - 3
                energy = np.sum(window**2)
                
                # row: which window are we in 
                row = (i * n_windows) + wind_idx 

                # Column: which group of features (channel) are we in?
                col_start = j * n_features
                col_end = col_start + n_features
                
                stat_desc[row, col_start:col_end] = [mean, std, rms, mx, skewness, kurtosis, energy]
                
                offset += n_samples
                wind_idx += 1
                
    return np.array(stat_desc), np.array(valence), np.array(arousal)

if __name__ == "__main__":
    data = sio.loadmat('s01.mat')
    stat_desc, valence, arousal = time_domain(data)
    print(stat_desc.shape)
    print(valence.shape)
    print(arousal.shape)