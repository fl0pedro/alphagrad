import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_agg_comparison(csv_path):
    df = pd.read_csv(csv_path)
    
    # Exclude non-sequence configs for cleaner comparison
    configs = sorted([c for c in df['Config'].unique() if c not in ['fwd', 'rev', 'jax_fwd', 'jax_rev', '1.0', 'efficient_100k']])
    if not configs:
        print("No sequence configs finished yet. Skipping plot.")
        return
    
    df_filtered = df[df['Config'].isin(configs)].copy()
    
    metrics = ['MSE', 'AbsSqErr', 'Frobenius', 'CosSim']
    methods = ['Flattened', 'Unweighted', 'WeightedNorm', 'WeightedVol']
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    for i, m_name in enumerate(metrics):
        plot_data = []
        for method in methods:
            col_name = f"{method}_{m_name}"
            if col_name in df_filtered.columns:
                data = df_filtered[col_name].dropna()
                for val in data:
                    plot_data.append({'Method': method, 'Value': val})
        
        if plot_data:
            df_plot = pd.DataFrame(plot_data)
            sns.boxplot(ax=axes[i], x='Method', y='Value', data=df_plot)
            axes[i].set_title(f'Aggregation Comparison: {m_name}')
            if m_name != 'CosSim':
                axes[i].set_yscale('log')
            axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('nn_grad_agg_comparison.png', dpi=200)
    print("Aggregation comparison plot saved to nn_grad_agg_comparison.png")

if __name__ == "__main__":
    plot_agg_comparison('nn_results_vmapped.csv')
