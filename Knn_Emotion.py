import numpy as np
import scipy.io as sio
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from time_domain import time_domain
from freq_domain import freq_domain


def run_full_analysis():
    subjects = ['s01', 's02', 's03']
    window_lengths = [0.5, 1.0, 2.0]
    k_values = range(1, 11)
    band_names = ["delta", "theta", "alpha", "beta", "gamma"]
    results = []
    # Calculate total iterations for the progress bar
    # 3 subjects * 3 windows * 8 feature types * 2 targets * 10 k-values
    total_tasks = len(subjects) * len(window_lengths) * 8 * 2 * len(k_values)
    pbar = tqdm(total=total_tasks, desc="Total Progress")

    for sub_id in subjects:
        try:
            data = sio.loadmat(f'{sub_id}.mat')
        except FileNotFoundError:
            print(f"\nWarning: {sub_id}.mat not found. Skipping.")
            pbar.update(len(window_lengths) * 8 * 2 * len(k_values))
            continue

        for n in window_lengths:
            #Get features
            time_feat, valence, arousal = time_domain(data, n=n)
            freq_feat_all, _, _ = freq_domain(data, n=n)

            #Define the different feature sets to test
            feature_sets = {
                'time_only': time_feat,
                'freq_only': freq_feat_all,
                'combined': np.concatenate([time_feat, freq_feat_all], axis=1)
            }
            #Extract individual bands
            for b_idx, b_name in enumerate(band_names):
                feature_sets[b_name] = freq_feat_all[:, b_idx::5]

            for feat_name, X in feature_sets.items():
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                for target_name, y in [("Valence", valence), ("Arousal", arousal)]:
                    for k in k_values:
                        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
                        scores = cross_val_score(knn, X_scaled, y, cv=5)
                        acc = np.mean(scores)

                        results.append({
                            'Subject': sub_id,
                            'N': n,
                            'Feature': feat_name,
                            'K': k,
                            'Target': target_name,
                            'Accuracy': acc
                        })
                        pbar.update(1)

    pbar.close()
    print("\n" + "=" * 50)
    print("FINAL RESULTS SUMMARY")
    print("=" * 50)

    for sub_id in subjects:
        for target in ["Valence", "Arousal"]:
            sub_res = [r for r in results if r['Subject'] == sub_id and r['Target'] == target]
            if not sub_res:
                continue
            best = max(sub_res, key=lambda x: x['Accuracy'])

            print(f"\nBest {target} for {sub_id}:")
            print(f"  Accuracy:      {best['Accuracy']:.2%}")
            print(f"  Segment (N):   {best['N']}s")
            print(f"  Feature Type:  {best['Feature']}")
            print(f"  Neighbors (K): {best['K']}")


if __name__ == "__main__":
    run_full_analysis()