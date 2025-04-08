import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import soundfile as sf
import statsmodels.api as sm
from scipy.signal import stft
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def calculate_sdr(references, estimates, eps=1e-7):
    num = np.sum(references ** 2)
    den = np.sum((references - estimates) ** 2)
    num += eps
    den += eps
    sdr = 10 * np.log10(num / den)
    return sdr


def calculate_si_sdr(references, estimates, eps=1e-7):
    scale = np.sum(estimates * references, axis=(0, 1)) / (np.sum(references ** 2, axis=(0, 1)) + eps)
    scale = np.expand_dims(scale, axis=(0, 1))
    scaled_references = references * scale
    sisdr = 10 * np.log10(
        np.sum(scaled_references ** 2, axis=(0, 1)) /
        (np.sum((scaled_references - estimates) ** 2, axis=(0, 1)) + eps)
    )
    return sisdr


def calculate_lsd(references, estimates, fs):
    _, _, Zxx_true = stft(references.T, fs=fs, window='hann', nperseg=1024, noverlap=512)
    _, _, Zxx_est = stft(estimates.T, fs=fs, window='hann', nperseg=1024, noverlap=512)
    SU_true = np.mean(np.abs(Zxx_true), axis=0)
    SU_est = np.mean(np.abs(Zxx_est), axis=0)
    epsilon = 1e-10
    SU_true = np.maximum(SU_true, epsilon)
    SU_est = np.maximum(SU_est, epsilon)
    N, K = SU_true.shape
    lsd = (1 / (N * K)) * np.sum(
        (10 * np.log10(SU_true / SU_est)) ** 2
    )
    return lsd


def run_one_way_anova(data, metric):
    model = ols(f'Q("{metric}") ~ C(System)', data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table, model


def visualize_data(data):
    metrics = ['SDR', 'SI-SDR', 'LSD']
    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.4)

    axes = [fig.add_subplot(gs[i]) for i in range(3)]

    for i, metric in enumerate(metrics):
        sns.boxplot(x='System', y=metric, hue='System', data=data, ax=axes[i], legend=False, palette='Set2')
        axes[i].set_title(f'Objective Metric: {metric}')
        axes[i].set_xlabel('System')
        axes[i].set_ylabel(metric)

    plt.tight_layout()
    plt.show()


def calculate_metrics(name_list, system_list, length="full", window_size=45, hop_size=5, silence_threshold=1e-4,
                      output_csv_path="data/metric_data.csv", df=None):
    results = []
    if df is None:
        for system in system_list:
            for name in name_list:
                reference_path = f"C:/Users/nickn/Desktop/test_stimuli/processed/dsp/{length}/{name}_ambience.wav"
                if system == "dnn":
                    estimate_path = f"C:/Users/nickn/Desktop/test_stimuli/processed/{system}/{length}/{name}_LR/ambience.wav"
                elif system == "mlp":
                    estimate_path = f"C:/Users/nickn/Desktop/test_stimuli/processed/{system}/{length}/{name}_LR_ambience.wav"

                references, fs = sf.read(reference_path)
                estimates, _ = sf.read(estimate_path)
                min_length = min(len(references), len(estimates))
                references = references[:min_length]
                estimates = estimates[:min_length]

                window_samples = int(window_size * fs)
                hop_samples = int(hop_size * fs)

                sdr_scores, sisdr_scores, lsd_scores, silent_frame_counts = [], [], [], 0
                total_frames = len(references)

                for start in range(0, total_frames - window_samples + 1, hop_samples):
                    ref_chunk = references[start:start + window_samples]
                    est_chunk = estimates[start:start + window_samples]

                    if np.max(np.abs(ref_chunk)) < silence_threshold:
                        silent_frame_counts += len(ref_chunk)
                        continue

                    sdr_scores.append(calculate_sdr(ref_chunk, est_chunk))
                    sisdr_scores.append(calculate_si_sdr(ref_chunk, est_chunk))
                    lsd_scores.append(calculate_lsd(ref_chunk, est_chunk, fs))

                if sdr_scores:
                    avg_sdr = np.mean(sdr_scores)
                    avg_sisdr = np.mean(sisdr_scores)
                    avg_lsd = np.mean(lsd_scores)
                    silence_percentage = (silent_frame_counts / total_frames) * 100
                    results.append([name, system, avg_sdr, avg_sisdr, avg_lsd, silence_percentage])
                    print(
                        f"{name} ({system}): SDR={avg_sdr:.2f} dB, SI-SDR={avg_sisdr:.2f} dB, LSD={avg_lsd:.2f} dB, Silence={silence_percentage:.2f}%")

        df = pd.DataFrame(results, columns=['Stimulus', 'System', 'SDR', 'SI-SDR', 'LSD', 'Silence %'])
    else:
        df.columns = ['Stimulus', 'System', 'SDR', 'SI-SDR', 'LSD', 'Silence %']

    # Print overall averages per system
    overall_metrics = df.groupby('System')[['SDR', 'SI-SDR', 'LSD', 'Silence %']].mean()
    print("\nOverall Average Metrics per System:")
    print(overall_metrics)

    if output_csv_path is not None:
        df.to_csv(output_csv_path, index=False)
        print(f"Cleaned metric data saved to {output_csv_path}")

    return df


def analyze_objective_metrics(data_path=None):
    name_list = [
        "alabama", "ballad", "escape", "five", "fred",
        "guardians", "jigsaw", "money", "queen", "stretch",
        "laufey", "snow"
    ]
    system_list = ["dnn", "mlp"]

    # Load or calculate metrics
    if data_path is None:
        data = calculate_metrics(name_list, system_list)
    else:
        data = pd.read_csv(data_path)
        data = calculate_metrics(name_list, system_list, df=data)

    # Perform one-way ANOVA for each metric
    for metric in ['SDR', 'SI-SDR', 'LSD']:
        print(f"\nAnalyzing {metric}...")
        anova_results, model = run_one_way_anova(data, metric)
        print(anova_results)

        # Check significance and run Tukey's HSD if needed
        system_pval = anova_results.loc["C(System)", "PR(>F)"]
        if system_pval < 0.05:
            print("System effect is significant; performing Tukey’s HSD post-hoc test:")
            tukey = pairwise_tukeyhsd(endog=data[metric], groups=data['System'], alpha=0.05)
            print(tukey)
        else:
            print("No significant system effect; no post-hoc test needed.")

    print("\nVisualizing data...")
    visualize_data(data)


if __name__ == "__main__":
    data_path = "data/metric_data.csv"
    analyze_objective_metrics(data_path)
