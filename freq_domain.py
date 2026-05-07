import numpy as np
import scipy.io as sio
from scipy.signal import welch
from scipy.integrate import trapezoid

bands = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}

def freq_domain(EEG: dict, n: float = 0.5) -> tuple:
    fs = EEG["fs"][0][0]
    n_samples = int(n * fs)
    n_trials = len(EEG["data"])
    n_channels = len(EEG["data"][0])
    n_windows = EEG["data"][0][0].shape[0] // n_samples
    n_features = len(bands)
    band_pow = np.zeros((n_trials * n_windows, n_channels * n_features))
    
    arousal = np.repeat([label[1] for label in EEG["labels"]], n_windows)
    valence = np.repeat([label[0] for label in EEG["labels"]], n_windows)
    
    # nperseg can't exceed the segment length
    nperseg = min(n_samples, 128)
    
    for i in range(n_trials):
        for j in range(n_channels):
            waves = EEG["data"][i][j]
            offset = 0
            wind_idx = 0
            
            while offset + n_samples <= waves.shape[0]:
                window = waves[offset:offset + n_samples]
                
                # PSD via Welch, then integrate over each band
                freqs, psd = welch(window, fs=fs, nperseg=nperseg)
                feats = []
                for low, high in bands.values():
                    mask = (freqs >= low) & (freqs <= high)
                    feats.append(trapezoid(psd[mask], freqs[mask]))
                
                row = (i * n_windows) + wind_idx
                col_start = j * n_features
                col_end = col_start + n_features
                
                band_pow[row, col_start:col_end] = feats
                
                offset += n_samples
                wind_idx += 1
                
    return np.array(band_pow), np.array(valence), np.array(arousal)

if __name__ == "__main__":
    data = sio.loadmat('s01.mat')
    band_pow, valence, arousal = freq_domain(data)
    print(band_pow.shape)
    print(valence.shape)
    print(arousal.shape)