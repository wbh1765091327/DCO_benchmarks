import numpy as np
import matplotlib.pyplot as plt
from utils import fvecs_read, fvecs_write
import os
import struct
from tqdm import tqdm
import pandas as pd
import matplotlib.ticker as ticker

datasets = ['glove-200-angular','deep-image-96-angular','contriever-768','instructorxl-arxiv-768','sift-128-euclidean','msong-420','gist-960-euclidean','openai-1536-angular']
ivf_marker = ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
col = ['r', 'b', 'y', 'g', 'c', 'm', 'olive', 'gold', 'violet', 'pink', 'brown', 'gray']

def load_result_data(filename, dataset):
    df = pd.read_csv(filename)
    df_dataset = df[df['dataset'] == dataset]
    recall = df_dataset['recall'].values / 100
    Qps = 1e6 / df_dataset['per_query_us'].values
    return recall, Qps

def load_result_data_RabitQ(filename, dataset):
    df = pd.read_csv(filename)
    df_dataset = df[df['dataset'] == dataset]
    recall = df_dataset['per_query_us'].values 
    Qps = df_dataset['recall'].values
    return recall, Qps

if __name__ == "__main__":
    for K in [10]:
        n_rows = 2
        n_cols = (len(datasets) + 1) // 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 8), sharex=False, sharey=False)

        fig.subplots_adjust(hspace=0.3, wspace=0.2)
        axes = axes.flatten()

        for idx, dataset in enumerate(datasets):
            ax = axes[idx]
            for i in range(5):
                if i == 0:
                    label = "hnsw"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Res/hnsw/nosimd/HNSW_RES_nosimd_0.csv"
                    recall, Qps = load_result_data(filename, dataset)
                    mask = recall >= 0.80
                    ax.plot(recall[mask], Qps[mask], marker=ivf_marker[i], c=col[i], label=label, alpha=0.5, linestyle="--", markerfacecolor='white', markersize=4)
                elif i == 1:
                    label = "hnsw-adsampling"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Res/hnsw/nosimd/HNSW_RES_nosimd_1.csv"
                    recall, Qps = load_result_data(filename, dataset)
                    mask = recall >= 0.80
                    ax.plot(recall[mask], Qps[mask], marker=ivf_marker[i], c=col[i], label=label, alpha=0.5, linestyle="--", markerfacecolor='white', markersize=4)
                elif i == 2:
                    label = "hnsw-bsa"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Res/hnsw/nosimd/HNSW_RES_nosimd_6.csv"
                    recall, Qps = load_result_data(filename, dataset)
                    mask = recall >= 0.80
                    ax.plot(recall[mask], Qps[mask], marker=ivf_marker[i], c=col[i], label=label, alpha=0.5, linestyle="--", markerfacecolor='white', markersize=4)
                elif i == 3:
                    label = "hnsw-dade"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/DADE/cleaned_HNSW_DADE_3.csv"
                    recall, Qps = load_result_data(filename, dataset)
                    mask = recall >= 0.80
                    ax.plot(recall[mask], Qps[mask], marker=ivf_marker[i], c=col[i], label=label, alpha=0.5, linestyle="--", markerfacecolor='white', markersize=4)
                elif i == 4:
                    label = "hnsw-finger"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Res/hnsw/nosimd/HNSW_RES_nosimd_9.csv"
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
        plt.savefig(f'E:/cppwork/dco_benchmarks/DATA/figure/ALL/HNSW/hnsw_qps_subplots_@{K}_nosimd.png', dpi=400, bbox_inches='tight')
        plt.show()

    for K in [10]:
        n_rows = 2
        n_cols = (len(datasets) + 1) // 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 8), sharex=False, sharey=False)

        fig.subplots_adjust(hspace=0.3, wspace=0.2)
        axes = axes.flatten()

        for idx, dataset in enumerate(datasets):
            ax = axes[idx]
            for i in range(6):
                if i == 0:
                    label = "hnsw"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Res/hnsw/avx512/HNSW_RES_avx512_0.csv"
                    recall, Qps = load_result_data(filename, dataset)
                    mask = recall >= 0.80
                    ax.plot(recall[mask], Qps[mask], marker=ivf_marker[i], c=col[i], label=label, alpha=0.5, linestyle="--", markerfacecolor='white', markersize=4)
                elif i == 1:
                    label = "hnsw-adsampling"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Res/hnsw/avx512/HNSW_RES_avx512_1.csv"
                    recall, Qps = load_result_data(filename, dataset)
                    mask = recall >= 0.80
                    ax.plot(recall[mask], Qps[mask], marker=ivf_marker[i], c=col[i], label=label, alpha=0.5, linestyle="--", markerfacecolor='white', markersize=4)
                elif i == 2:
                    label = "hnsw-bsa"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Res/hnsw/avx512/HNSW_RES_avx512_6.csv"
                    recall, Qps = load_result_data(filename, dataset)
                    mask = recall >= 0.80
                    ax.plot(recall[mask], Qps[mask], marker=ivf_marker[i], c=col[i], label=label, alpha=0.5, linestyle="--", markerfacecolor='white', markersize=4)
                elif i == 3:
                    label = "hnsw-dade"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/DADE/cleaned_HNSW_DADE_3_SIMD.csv"
                    recall, Qps = load_result_data(filename, dataset)
                    mask = recall >= 0.80
                    ax.plot(recall[mask], Qps[mask], marker=ivf_marker[i], c=col[i], label=label, alpha=0.5, linestyle="--", markerfacecolor='white', markersize=4)
                elif i == 4:
                    label = "hnsw-finger"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Res/hnsw/avx512/HNSW_RES_avx512_9.csv"
                    recall, Qps = load_result_data(filename, dataset)
                    mask = recall >= 0.80
                    ax.plot(recall[mask], Qps[mask], marker=ivf_marker[i], c=col[i], label=label, alpha=0.5, linestyle="--", markerfacecolor='white', markersize=4)
                # elif i == 5:
                #     label = "hnsw-RaBitQ"
                #     filename = f"E:/cppwork/dco_benchmarks/DATA/RabitQ/HNSW_RabitQ.csv"
                #     recall, Qps = load_result_data_RabitQ(filename, dataset)
                #     mask = recall >= 0.80
                #     ax.plot(recall[mask], Qps[mask], marker=ivf_marker[i], c=col[i], label=label, alpha=0.5, linestyle="--", markerfacecolor='white', markersize=4)

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
        plt.savefig(f'E:/cppwork/dco_benchmarks/DATA/figure/ALL/HNSW/hnsw_qps_subplots_@{K}_simd.png', dpi=400, bbox_inches='tight')
        plt.show()
