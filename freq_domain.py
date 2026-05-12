import numpy as np
import scipy.io as sio

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

    for i in range(n_trials):
        for j in range(n_channels):
            waves = EEG["data"][i][j]

            offset = 0
            wind_idx = 0

            while offset + n_samples <= waves.shape[0]:
                window = waves[offset:offset + n_samples]
                window = window - np.mean(window)

                fft_values = np.fft.rfft(window)
                freqs = np.fft.rfftfreq(n_samples, d=1/fs)
                power_spectrum = (np.abs(fft_values) ** 2) / n_samples

                feats = []

                for band_idx, (low, high) in enumerate(bands.values()):
                    if band_idx == len(bands) - 1:
                        mask = (freqs >= low) & (freqs <= high)
                    else:
                        mask = (freqs >= low) & (freqs < high)

                    band_power = np.sum(power_spectrum[mask])
                    feats.append(band_power)

                row = (i * n_windows) + wind_idx
                col_start = j * n_features
                col_end = col_start + n_features

                band_pow[row, col_start:col_end] = feats

                offset += n_samples
                wind_idx += 1

    return np.array(band_pow), np.array(valence), np.array(arousal)


if __name__ == "__main__":
    data = sio.loadmat('D:\AUC\Spring 2026\DSP\Project\Data\s01.mat')#make sure to change this path to the correct one 
    band_pow, valence, arousal = freq_domain(data)

    print("Frequency features:", band_pow.shape)
    print("Valence labels:", valence.shape)
    print("Arousal labels:", arousal.shape)