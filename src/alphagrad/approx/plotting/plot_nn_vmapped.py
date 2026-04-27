import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_results(csv_path):
    df = pd.read_csv(csv_path)
    
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    
    # Consistent palette
    config_list = sorted(df["Config"].unique())
    palette = dict(zip(config_list, sns.color_palette("husl", len(config_list))))
    
    # Mode filtering for consistency
    if "Mode" in df.columns:
        df_time = df[df["Mode"] == "time"].copy()
        df_epochs = df[df["Mode"] == "epochs"].copy()
        df_eff = df[df["Mode"] == "efficient"].copy()
    else:
        df_time = df.copy()
        df_epochs = df.copy()
        df_eff = pd.DataFrame()

    df_time["TotalTimeSec"] = df_time["TotalLatency"] / 1000.0
    df_epochs["TotalTimeSec"] = df_epochs["TotalLatency"] / 1000.0
    if not df_eff.empty:
        df_eff["TotalTimeSec"] = df_eff["TotalLatency"] / 1000.0

    # First row: vs Epoch (Original modes)
    sns.lineplot(ax=axes[0, 0], data=df_epochs, x="Epoch", y="Loss", hue="Config", palette=palette)
    axes[0, 0].set_title("Training Loss vs Epoch")
    axes[0, 0].set_yscale("log")
    
    sns.lineplot(ax=axes[0, 1], data=df_epochs, x="Epoch", y="ValAcc", hue="Config", palette=palette)
    axes[0, 1].set_title("Validation Accuracy vs Epoch")
    axes[0, 1].set_ylim(0, 1.05)
    
    sns.lineplot(ax=axes[0, 2], data=df_epochs, x="Epoch", y="TotalLatency", hue="Config", palette=palette)
    axes[0, 2].set_title("Cumulative Latency (ms) vs Epoch")
    
    # Second row: over Time (Original modes)
    sns.lineplot(ax=axes[1, 0], data=df_time, x="TotalTimeSec", y="Loss", hue="Config", palette=palette)
    axes[1, 0].set_title("Training Loss over Time")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("Time (s)")
    
    sns.lineplot(ax=axes[1, 1], data=df_time, x="TotalTimeSec", y="ValAcc", hue="Config", palette=palette)
    axes[1, 1].set_title("Validation Accuracy over Time")
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_xlabel("Time (s)")
    
    # Zoomed version in the last subplot of row 2
    df_zoom = df_time[(df_time["TotalTimeSec"] <= 2.0) & (df_time["ValAcc"] >= 0.8)].copy()
    zoom_configs = df_zoom["Config"].unique()
    df_plot_zoom = df_time[df_time["Config"].isin(zoom_configs)].copy()
    sns.lineplot(ax=axes[1, 2], data=df_plot_zoom, x="TotalTimeSec", y="ValAcc", hue="Config", palette=palette)
    axes[1, 2].set_title("Zoomed Val Acc over Time (0-2s, 0.8-1.0)")
    axes[1, 2].set_xlim(0, 2.0)
    axes[1, 2].set_ylim(0.8, 1.02)
    axes[1, 2].set_xlabel("Time (s)")

    # Third row: Efficient Mode
    if not df_eff.empty:
        sns.lineplot(ax=axes[2, 0], data=df_eff, x="TotalTimeSec", y="Loss", hue="Config", palette=palette)
        axes[2, 0].set_title("Efficient Mode: Loss over Time")
        axes[2, 0].set_yscale("log")
        axes[2, 0].set_xlabel("Time (s)")

        sns.lineplot(ax=axes[2, 1], data=df_eff, x="TotalTimeSec", y="ValAcc", hue="Config", palette=palette)
        axes[2, 1].set_title("Efficient Mode: Val Acc over Time")
        axes[2, 1].set_ylim(0, 1.05)
        axes[2, 1].set_xlabel("Time (s)")

        # Zoomed Efficient Mode (First 1s)
        sns.lineplot(ax=axes[2, 2], data=df_eff, x="TotalTimeSec", y="ValAcc", hue="Config", palette=palette)
        axes[2, 2].set_title("Efficient Mode: Zoomed (0-1s)")
        axes[2, 2].set_xlim(0, 1.0)
        axes[2, 2].set_ylim(0.5, 1.05)
        axes[2, 2].set_xlabel("Time (s)")
    else:
        for j in range(3):
            axes[2, j].text(0.5, 0.5, "No Efficient Data", ha='center', va='center')

    plt.tight_layout()
    plt.savefig("nn_results_vmapped.png", dpi=300)
    print("Plot saved to nn_results_vmapped.png")

if __name__ == "__main__":
    plot_results("nn_results_vmapped.csv")
