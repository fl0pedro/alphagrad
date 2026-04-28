import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
df = pd.read_csv('nn_results_vmapped.csv')

# Metrics and Variables to plot
flattened_metrics = ['Flattened_MSE', 'Flattened_AbsSqErr', 'Flattened_Frobenius', 'Flattened_CosSim']
per_var_metrics = ['MSE', 'AbsSqErr', 'Frobenius', 'CosSim']
variables = ['W1', 'b1', 'W2', 'b2']
configs = sorted([c for c in df['Config'].unique() if c not in ['fwd', 'rev', 'jax_fwd', 'jax_rev', '1.0', 'efficient_100k']])

if not configs:
    print("No sequence configs finished yet. Skipping plot.")
else:
    # Per-variable analysis
    fig, axes = plt.subplots(len(per_var_metrics), len(variables), figsize=(24, 20))
    for i, metric in enumerate(per_var_metrics):
        for j, var in enumerate(variables):
            col_name = f"{metric}_{var}"
            if col_name not in df.columns:
                continue
                
            data_to_plot = []
            for config in configs:
                data = df[df['Config'] == config][col_name].dropna()
                data_to_plot.append(data)
            
            axes[i, j].boxplot(data_to_plot, labels=configs)
            axes[i, j].set_title(f'{var} - {metric}')
            if metric != 'CosSim':
                all_pos = all((d > 0).all() for d in data_to_plot if len(d) > 0)
                if all_pos:
                    axes[i, j].set_yscale('log')
            axes[i, j].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('nn_grad_analysis_per_var.png', dpi=200)
    print("Per-variable plot saved to nn_grad_analysis_per_var.png")

    # Overall (Flattened) analysis
    fig2, axes2 = plt.subplots(1, len(flattened_metrics), figsize=(24, 6))
    for i, metric in enumerate(flattened_metrics):
        if metric not in df.columns:
            continue
        data_to_plot = []
        for config in configs:
            data = df[df['Config'] == config][metric].dropna()
            data_to_plot.append(data)
        
        axes2[i].boxplot(data_to_plot, labels=configs)
        axes2[i].set_title(f'Overall {metric}')
        if 'CosSim' not in metric:
            all_pos = all((d > 0).all() for d in data_to_plot if len(d) > 0)
            if all_pos:
                axes2[i].set_yscale('log')
        axes2[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('nn_grad_analysis_flattened.png', dpi=200)
    print("Flattened plot saved to nn_grad_analysis_flattened.png")
