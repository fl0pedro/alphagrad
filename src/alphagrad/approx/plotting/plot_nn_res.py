import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Plot training results from CSV')
    parser.add_argument('file', help='Path to the CSV file')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--estimated', action='store_true', help='Use estimated lines')
    group.add_argument('--analytical', action='store_true', help='Use analytical lines')
    group.add_argument('--empirical', action='store_true', help='Empirical results (no lines for now)')
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File {args.file} not found.")
        return

    df = pd.read_csv(args.file, skipinitialspace=True)
    
    title = "Estimated" if args.estimated else "Analytical" if args.analytical else "Empirical"
    
    # 2 rows, 4 columns layout
    fig, axes = plt.subplots(2, 4, figsize=(25.6, 8), sharex=False)
    fig.suptitle(title, fontsize=20)
    
    # Column mapping: 0: Entropy/Empty, 1: CMP, 2: Mem, 3: Acc
    ax_ent = axes[0, 0]
    ax_bcmp = axes[0, 1]
    ax_bmem = axes[0, 2]
    ax_bacc = axes[0, 3]
    ax_empty = axes[1, 0]
    ax_mcmp = axes[1, 1]
    ax_mmem = axes[1, 2]
    ax_macc = axes[1, 3]

    # Plot Entropy
    if 'ent' in df.columns:
        df['ent'].plot(ax=ax_ent, color='tab:blue')
        ax_ent.set_title('Entropy')
        ax_ent.set_ylabel('Value')
        ax_ent.grid(True, alpha=0.3)

    # Plot Best Metrics
    if 'best_cmp' in df.columns:
        df['best_cmp'].plot(ax=ax_bcmp, color='tab:green')
        ax_bcmp.set_title('Best CMP (Cost)')
        ax_bcmp.set_ylabel('Value')
        ax_bcmp.grid(True, alpha=0.3)
        if args.estimated:
            ax_bcmp.axhline(y=16576.0, color='red', linestyle='--', label='GPU fwd', alpha=0.5)
            ax_bcmp.axhline(y=9280.0, color='magenta', linestyle='--', label='GPU rev', alpha=0.5)
            ax_bcmp.axhline(y=16192.0, color='blue', linestyle=':', label='CPU fwd', alpha=0.5)
            ax_bcmp.axhline(y=8896.0, color='cyan', linestyle=':', label='CPU rev', alpha=0.5)
            ax_bcmp.legend(fontsize='x-small')
        elif args.analytical:
            ax_bcmp.axhline(y=38431.0, color='red', linestyle='--', label='fwd', alpha=0.5)
            ax_bcmp.axhline(y=9829.0, color='magenta', linestyle='--', label='rev', alpha=0.5)
            ax_bcmp.legend(fontsize='small')

    if 'best_mem' in df.columns:
        df['best_mem'].plot(ax=ax_bmem, color='tab:green')
        ax_bmem.set_title('Best Memory (Bytes)')
        ax_bmem.set_ylabel('Value')
        ax_bmem.grid(True, alpha=0.3)
        if args.estimated:
            ax_bmem.axhline(y=43952.0, color='black', linestyle='--', label='GPU fwd/rev', alpha=0.5)
            ax_bmem.axhline(y=72632.0, color='grey', linestyle=':', label='CPU fwd', alpha=0.5)
            ax_bmem.axhline(y=89016.0, color='brown', linestyle=':', label='CPU rev', alpha=0.5)
            ax_bmem.legend(fontsize='x-small')
        elif args.analytical:
            ax_bmem.axhline(y=77312.0, color='black', linestyle='--', label='fwd', alpha=0.5)
            ax_bmem.axhline(y=19712.0, color='grey', linestyle='--', label='rev', alpha=0.5)
            ax_bmem.legend(fontsize='small')

    if 'best_acc' in df.columns:
        df['best_acc'].plot(ax=ax_bacc, color='tab:green')
        ax_bacc.set_title('Best Accuracy')
        ax_bacc.set_ylabel('Value')
        ax_bacc.grid(True, alpha=0.3)

    # Plot Mean Metrics
    if 'mean_cmp' in df.columns:
        df['mean_cmp'].plot(ax=ax_mcmp, color='tab:orange')
        ax_mcmp.set_title('Mean CMP (Cost)')
        ax_mcmp.set_ylabel('Value')
        ax_mcmp.grid(True, alpha=0.3)
        if args.estimated:
            ax_mcmp.axhline(y=16576.0, color='red', linestyle='--', label='GPU fwd', alpha=0.5)
            ax_mcmp.axhline(y=9280.0, color='magenta', linestyle='--', label='GPU rev', alpha=0.5)
            ax_mcmp.axhline(y=16192.0, color='blue', linestyle=':', label='CPU fwd', alpha=0.5)
            ax_mcmp.axhline(y=8896.0, color='cyan', linestyle=':', label='CPU rev', alpha=0.5)
            ax_mcmp.legend(fontsize='x-small')
        elif args.analytical:
            ax_mcmp.axhline(y=38431.0, color='red', linestyle='--', label='fwd', alpha=0.5)
            ax_mcmp.axhline(y=9829.0, color='magenta', linestyle='--', label='rev', alpha=0.5)
            ax_mcmp.legend(fontsize='small')

    if 'mean_mem' in df.columns:
        df['mean_mem'].plot(ax=ax_mmem, color='tab:orange')
        ax_mmem.set_title('Mean Memory (Bytes)')
        ax_mmem.set_ylabel('Value')
        ax_mmem.grid(True, alpha=0.3)
        if args.estimated:
            ax_mmem.axhline(y=43952.0, color='black', linestyle='--', label='GPU fwd/rev', alpha=0.5)
            ax_mmem.axhline(y=72632.0, color='grey', linestyle=':', label='CPU fwd', alpha=0.5)
            ax_mmem.axhline(y=89016.0, color='brown', linestyle=':', label='CPU rev', alpha=0.5)
            ax_mmem.legend(fontsize='x-small')
        elif args.analytical:
            ax_mmem.axhline(y=77312.0, color='black', linestyle='--', label='fwd', alpha=0.5)
            ax_mmem.axhline(y=19712.0, color='grey', linestyle='--', label='rev', alpha=0.5)
            ax_mmem.legend(fontsize='small')

    if 'mean_acc' in df.columns:
        df['mean_acc'].plot(ax=ax_macc, color='tab:orange')
        ax_macc.set_title('Mean Accuracy')
        ax_macc.set_ylabel('Value')
        ax_macc.grid(True, alpha=0.3)

    # Hide unused subplot
    ax_empty.set_visible(False)

    plt.xlabel('Episode / Step')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_png = os.path.splitext(args.file)[0] + '.png'
    plt.savefig(output_png, dpi=200)
    print(f"Saved {title} plot to {output_png}")

if __name__ == "__main__":
    main()