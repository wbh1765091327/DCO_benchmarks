import numpy as np
import matplotlib.pyplot as plt
from utils import fvecs_read, fvecs_write
import os
import struct
from tqdm import tqdm
import pandas as pd
import matplotlib.ticker as ticker


datasets = ['glove-200-angular','deep-image-96-angular','contriever-768','instructorxl-arxiv-768','sift-128-euclidean','msong-420','gist-960-euclidean','openai-1536-angular']

datasets = ['glove-200-angular','deep-image-96-angular','contriever-768','instructorxl-arxiv-768','sift-128-euclidean','msong-420','gist-960-euclidean','openai-1536-angular']
ivf_marker = ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
col = ['#FF0000', '#0000FF', '#00AA00', '#FF8000', '#8000FF', '#FF1493', '#008B8B', '#B8860B', '#4B0082', '#DC143C', '#228B22', '#FF4500']

def load_result_data(filename, dataset):
    df = pd.read_csv(filename)
    # Filter data for specific dataset
    df_dataset = df[df['dataset'] == dataset]
    # Get recall and QPS data
    recall = df_dataset['recall'].values
    # Calculate QPS (queries per second)
    Qps = 1e3 / df_dataset['avg_all'].values  # Convert average query time (ms) to QPS
    return recall, Qps


def format_sci(x, pos, exp):
    if x == 0:
        return '0'
    try:
        coef = x / 10**exp
        return f'${coef:.1f} \\times 10^{{{exp}}}$'
    except:
        return '0'


if __name__ == "__main__":
    for K in [10]:
        n_rows = 2
        n_cols = (len(datasets) + 1) // 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 8), sharex=False, sharey=False)

        fig.subplots_adjust(hspace=0.3, wspace=0.2)
        axes = axes.flatten()

        for idx, dataset in enumerate(datasets):
            ax = axes[idx]
            for i in range(4):
                filename = f"E:/cppwork/dco_benchmarks/DATA/PDX/pdx-bond.csv"
                label = "null"
                if i == 0:
                    label = "ivf-pdx-adsampling_decimpr"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/PDXnew/split_by_algorithm/pdx_adsampling_decimpr.csv"
                elif i == 1:
                    label = "ivf-pdx-adsampling_dz"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/PDXnew/split_by_algorithm/pdx_adsampling_dz.csv"
                elif i == 2:
                    label = "ivf-pdx-adsampling"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/PDXnew/split_by_algorithm/pdx_adsampling.csv"
                elif i == 3:
                    label = "ivf-pdx-adsampling_sec"       
                    filename = f"E:/cppwork/dco_benchmarks/DATA/PDX/IVF_PDX_ADSAMPLING.csv"


                recall, Qps = load_result_data(filename, dataset)
                mask = recall >= 0.80
                ax.plot(recall[mask], Qps[mask], marker=ivf_marker[i], c=col[i], label=label, alpha=0.5, linestyle="--", markerfacecolor='white', markersize=4)

            ax.set_title(dataset, fontsize=10)
            ax.grid(linestyle='--', linewidth=0.5)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
            ax.set_ylim(bottom=0)
            ax.set_xlim(left=0.80)
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))

        # 设置公共x/y轴标签
        fig.text(0.5, 0.04, 'Recall@10', ha='center', fontsize=14)
        fig.text(0.06, 0.5, 'Qps', va='center', rotation='vertical', fontsize=14)

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=10)

        plt.rc('font', family='Times New Roman')
        plt.savefig(f'E:/cppwork/dco_benchmarks/DATA/figure/BOND/ivf_dimorder_ads.png', dpi=400)
        plt.show()

    for K in [10]:
        n_rows = 2
        n_cols = (len(datasets) + 1) // 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 8), sharex=False, sharey=False)

        fig.subplots_adjust(hspace=0.3, wspace=0.2)
        axes = axes.flatten()

        for idx, dataset in enumerate(datasets):
            ax = axes[idx]
            for i in range(4):
                filename = f"E:/cppwork/dco_benchmarks/DATA/PDX/pdx-bond.csv"
                label = "null"
                if i == 0:
                    label = "ivf-pdx-bsa_decimpr"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/PDXnew/split_by_algorithm/pdx_bsa_decimpr.csv"
                elif i == 1:
                    label = "ivf-pdx-bsa_dz"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/PDXnew/split_by_algorithm/pdx_bsa_dz.csv"
                elif i == 2:
                    label = "ivf-pdx-bsa"       
                    filename = f"E:/cppwork/dco_benchmarks/DATA/PDXnew/split_by_algorithm/pdx_bsa.csv"
                elif i == 3:
                    label = "ivf-pdx-bsa_sec"       
                    filename = f"E:/cppwork/dco_benchmarks/DATA/PDX/IVF_PDX_BSA.csv"

                recall, Qps = load_result_data(filename, dataset)
                mask = recall >= 0.80
                ax.plot(recall[mask], Qps[mask], marker=ivf_marker[i], c=col[i], label=label, alpha=0.5, linestyle="--", markerfacecolor='white', markersize=4)

            ax.set_title(dataset, fontsize=10)
            ax.grid(linestyle='--', linewidth=0.5)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
            ax.set_ylim(bottom=0)
            ax.set_xlim(left=0.80)
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))

        # 设置公共x/y轴标签
        fig.text(0.5, 0.04, 'Recall@10', ha='center', fontsize=14)
        fig.text(0.06, 0.5, 'Qps', va='center', rotation='vertical', fontsize=14)

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=10)

        plt.rc('font', family='Times New Roman')
        plt.savefig(f'E:/cppwork/dco_benchmarks/DATA/figure/BOND/ivf_dimorder_bsa.png', dpi=400)
        plt.show()