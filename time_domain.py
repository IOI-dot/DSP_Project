import scipy.io as spio
import numpy as np
def time_domain( EEG: dict, n: float = 0.5 ) -> tuple:
    fs = EEG["fs"][0][0]
    n_samples = int(n * fs)
    n_trials = len(EEG["data"])
    n_channels = len(EEG["data"][0])
    n_windows = EEG["data"][0][0].shape[0] // n_samples
    n_features = 7
    
    arousal = np.zeros(n_trials)
    valence = np.zeros(n_trials)
    stat_desc = np.zeros((n_trials, n_channels, n_windows, n_features))
    for i, trial in enumerate(EEG["data"]):
        valence[i] = EEG["labels"][i][0]
        arousal[i] = EEG["labels"][i][1]
        for j,waves in enumerate(trial):
            offset = 0
            wind_idx = 0
            print(f"Processing trial {i}, channel {j}...")
            while offset + n_samples <= waves.shape[0]:
                window = EEG["data"][i][j][offset:offset + n_samples]
                mean = window.mean()
                std = window.std()
                rms = np.sqrt(np.mean(window**2))
                max = np.max(window)
                skewness = (np.mean((window - mean)**3)) / (std**3)
                kurtosis = (np.mean((window - mean)**4)) / (std**4) - 3
                energy = np.sum(window**2)
                offset += n_samples
                stat_desc[i][j][wind_idx] = [mean, std, rms, max, skewness, kurtosis, energy]
                wind_idx += 1
    return stat_desc, valence, arousal  