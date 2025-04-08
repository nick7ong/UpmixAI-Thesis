import matplotlib.pyplot as plt
import pandas as pd
import scikit_posthocs as sp
import seaborn as sns
from scipy.stats import friedmanchisquare


def load_data(file_path):
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Use CSV or Excel.")


def run_friedman_test(data, dimension):
    for stim in data['Stimulus'].unique():
        print(f"\nFriedman Test for Stimulus: {stim}")
        subset = data[data['Stimulus'] == stim]
        pivot = subset.pivot(index='Participant', columns='System', values=dimension)

        if pivot.isnull().values.any():
            print("Skipping due to missing data")
            continue

        stat, p = friedmanchisquare(*[pivot[sys] for sys in pivot.columns])
        print(f"Friedman chi-square = {stat:.3f}, p = {p:.4f}")

        if p < 0.05:
            friedman_posthoc_nemenyi(pivot, stim)


def friedman_posthoc_nemenyi(pivot, label):
    p_values = sp.posthoc_nemenyi_friedman(pivot)
    print(f"\nNemenyi post-hoc test for {label}")
    print(p_values)


def run_friedman_test_on_overall_metrics(data):
    long_data = pd.melt(
        data,
        id_vars=['Participant', 'System'],
        value_vars=[
            'Overall Envelopment & Immersion',
            'Spatial & Temporal Quality',
            'Spectral Quality'
        ],
        var_name='Metric',
        value_name='Rating'
    )

    print("\n=== Friedman Test on Overall Metrics ===")
    for metric in long_data['Metric'].unique():
        subset = long_data[long_data['Metric'] == metric]

        collapsed = (
            subset.groupby(['Participant', 'System'])['Rating']
            .mean()
            .reset_index()
        )

        pivot = collapsed.pivot(index='Participant', columns='System', values='Rating')

        if pivot.isnull().values.any():
            print(f"Skipping {metric} due to missing data.")
            continue

        stat, p = friedmanchisquare(*[pivot[sys] for sys in pivot.columns])
        print(f"{metric}: chi-square = {stat:.3f}, p = {p:.4f}")

        if p < 0.05:
            friedman_posthoc_nemenyi(pivot, metric)


def visualize_data(data, dimension):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Stimulus', y=dimension, hue='System', data=data, palette='Set2')
    plt.title(f'Friedman Test Ratings for {dimension}')
    plt.xlabel('Stimulus')
    plt.ylabel(dimension)
    plt.legend(
        title='System',
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        frameon=False
    )
    plt.tight_layout()
    plt.show()


def reshape_and_visualize_overall_metrics(data):
    long_data = pd.melt(
        data,
        id_vars=['Participant', 'System'],
        value_vars=[
            'Overall Envelopment & Immersion',
            'Spatial & Temporal Quality',
            'Spectral Quality'
        ],
        var_name='Metric',
        value_name='Rating'
    )

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Metric', y='Rating', hue='System', data=long_data, palette='Set2')
    plt.title('Overall Ratings per System Across All Perceptual Metrics')
    plt.xlabel('Perceptual Metric')
    plt.ylabel('Rating')
    plt.legend(
        title='System',
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        frameon=False
    )
    plt.tight_layout()
    plt.show()


def print_overall_system_ranking(data, dimension):
    print(f"\n=== Overall System Ranking for {dimension} ===")
    overall = data.groupby('System')[dimension].mean().sort_values(ascending=False)
    print(overall)
    return overall


def analyze_mushra_friedman(file_path, visualize=True):
    data = load_data(file_path)
    dimensions = [
        'Overall Envelopment & Immersion',
        'Spatial & Temporal Quality',
        'Spectral Quality'
    ]

    overall_scores = pd.DataFrame()
    for dimension in dimensions:
        print(f"\n=== Analyzing {dimension} ===")
        run_friedman_test(data, dimension)
        scores = print_overall_system_ranking(data, dimension)
        overall_scores[dimension] = scores
        if visualize:
            visualize_data(data, dimension)

    print("\n=== Overall Combined Mean Ratings Across All Metrics ===")
    overall_scores["Total Mean"] = overall_scores.sum(axis=1)
    print(overall_scores.sort_values("Total Mean", ascending=False))

    run_friedman_test_on_overall_metrics(data)

    if visualize:
        reshape_and_visualize_overall_metrics(data)


if __name__ == "__main__":
    file_path = "data/cleaned_mushra_data.csv"
    analyze_mushra_friedman(file_path, visualize=False)
