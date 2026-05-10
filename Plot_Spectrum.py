import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import welch


def generate_psd_plots(subject_file='s01.mat'):
    try:
        data = sio.loadmat(subject_file)
    except FileNotFoundError:
        print(f"Error: {subject_file} not found. Please ensure it's in the same folder.")
        return

    fs = data['fs'][0][0]

    channel_idx = 14

    # Access the channel name
    try:
        raw_name = data['channel_names'][channel_idx]
        channel_name = str(raw_name[0]).strip() if isinstance(raw_name, np.ndarray) else str(raw_name).strip()
    except:
        channel_name = f"Channel {channel_idx}"

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Define targets and their column indices
    targets = [("Valence", 0), ("Arousal", 1)]

    for ax, (target_name, col_idx) in zip(axes, targets):
        # Get labels for the current target
        labels = data['labels'][:, col_idx]
        high_trials = np.where(labels == 1)[0]
        low_trials = np.where(labels == 0)[0]

        if len(high_trials) == 0 or len(low_trials) == 0:
            print(f"Error: Could not find both High and Low {target_name} trials.")
            continue

        high_idx = high_trials[0]
        low_idx = low_trials[0]

        for idx, label_text in zip([high_idx, low_idx], [f"High {target_name}", f"Low {target_name}"]):
            # Get signal for the chosen channel (Trial x Channel x Samples)
            signal = data['data'][idx][channel_idx]

            # Calculate PSD using Welch method
            freqs, psd = welch(signal, fs=fs, nperseg=min(len(signal), 256))

            ax.plot(freqs, psd, label=label_text, linewidth=2)

        # Formatting
        ax.set_title(f"PSD Comparison for {target_name}")
        ax.set_xlabel("Frequency (Hz)")
        if target_name == "Valence":
            ax.set_ylabel("Power Spectral Density (V^2/Hz)")
        ax.set_xlim(0, 45)  # Focus on EEG range
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, which="both", ls="-", alpha=0.5)

    # Add a main title
    plt.suptitle(f"Spectral Analysis for {channel_name} (Subject 01)", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('psd_plots_combined.png', bbox_inches='tight')
    plt.show()
    print("Success! Plots saved as 'psd_plots_combined.png'.")


if __name__ == "__main__":
    generate_psd_plots('s01.mat')