import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, ttest_rel, wilcoxon

from datasets import load_dataset, train_test_split_graph
from gnn_main import train_and_evaluate  # assumes you have a training function that returns F1


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_experiments(dataset_name="amazon", gnn_type="GraphSAGE", runs=30):
    baseline_f1, enriched_f1 = [], []

    for seed in range(runs):
        set_seed(seed)

        # --- Baseline dataset ---
        g = load_dataset(dataset_name, enriched=False)
        train_mask, val_mask, test_mask = train_test_split_graph(g, random_state=seed)
        f1_base = train_and_evaluate(g, gnn_type, train_mask, val_mask, test_mask)
        print(f"Run {seed+1}/{runs} - Baseline F1: {f1_base:.4f}")
        baseline_f1.append(f1_base)

        # --- Enriched dataset ---
        g_en = load_dataset(dataset_name, enriched=True, load_path =f"enriched_graph_{dataset_name}.dgl")
        train_mask, val_mask, test_mask = train_test_split_graph(g_en, random_state=seed)
        f1_en = train_and_evaluate(g_en, gnn_type, train_mask, val_mask, test_mask)
        print(f"Run {seed+1}/{runs} - Enriched F1: {f1_en:.4f}")
        enriched_f1.append(f1_en)

    return np.array(baseline_f1), np.array(enriched_f1)


def statistical_test(baseline, enriched):
    # Normality test
    _, p_base = shapiro(baseline)
    _, p_en = shapiro(enriched)

    if p_base > 0.05 and p_en > 0.05:
        # Both distributions look normal -> paired t-test
        stat, p_val = ttest_rel(baseline, enriched)
        test_name = "Paired t-test"
    else:
        # Non-parametric alternative
        stat, p_val = wilcoxon(baseline, enriched)
        test_name = "Wilcoxon signed-rank"

    return test_name, stat, p_val


def plot_results(baseline, enriched, dataset_name, gnn_type):
    plt.figure(figsize=(7,5))
    sns.boxplot(data=[baseline, enriched], palette="Set2")
    sns.stripplot(data=[baseline, enriched], color="black", alpha=0.5)

    plt.xticks([0,1], ["Baseline", "Enriched"])
    plt.ylabel("F1 score")
    plt.title(f"{gnn_type} on {dataset_name} - Baseline vs Enriched Features")
    plt.show()


if __name__ == "__main__":
    dataset = "yelp"
    gnn_type = "GraphSAGE"
    runs = 30

    baseline, enriched = run_experiments(dataset, gnn_type, runs)
    print(f"Baseline mean ± std: {baseline.mean():.4f} ± {baseline.std():.4f}")
    print(f"Enriched mean ± std: {enriched.mean():.4f} ± {enriched.std():.4f}")

    test_name, stat, p_val = statistical_test(baseline, enriched)
    print(f"{test_name}: statistic={stat:.4f}, p={p_val:.4f}")

    plot_results(baseline, enriched, dataset, gnn_type)