import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_agg_grid(csv_path):
    df = pd.read_csv(csv_path)
    
    # Filter for sequence configs in range [0, 1)
    # Exclude standard benchmarks and the 1.0 config
    excluded = ['fwd', 'rev', 'jax_fwd', 'jax_rev', '1.0', 'efficient_100k']
    configs = sorted([c for c in df['Config'].unique() if c not in excluded])
    
    if not configs:
        print("No sequence configs finished yet. Skipping plot.")
        return
    
    metrics = ['MSE', 'AbsSqErr', 'Frobenius', 'CosSim']
    methods = ['Flattened', 'Unweighted', 'WeightedNorm', 'WeightedVol']
    
    fig, axes = plt.subplots(len(metrics), len(methods), figsize=(30, 24))
    
    for i, metric in enumerate(metrics):
        for j, method in enumerate(methods):
            col_name = f"{method}_{metric}"
            if col_name not in df.columns:
                axes[i, j].text(0.5, 0.5, f"Column {col_name}\nnot found", 
                               ha='center', va='center')
                continue
                
            data_to_plot = []
            for config in configs:
                data = df[df['Config'] == config][col_name].dropna()
                data_to_plot.append(data)
            
            axes[i, j].boxplot(data_to_plot, tick_labels=configs)
            axes[i, j].set_title(f'{method} {metric}')
            
            if 'CosSim' not in metric:
                # Check for positive values for log scale
                all_pos = all((d > 0).all() for d in data_to_plot if len(d) > 0)
                if all_pos:
                    axes[i, j].set_yscale('log')
            
            axes[i, j].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('nn_grad_agg_grid.png', dpi=200)
    print("Aggregation grid plot saved to nn_grad_agg_grid.png")

if __name__ == "__main__":
    plot_agg_grid('nn_results_vmapped.csv')
