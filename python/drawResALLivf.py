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
    
    # 创建DataFrame来处理重复的recall值
    data_df = pd.DataFrame({'recall': recall, 'qps': Qps})
    
    # 对于相同的recall值，取最小的qps值
    data_df = data_df.groupby('recall')['qps'].min().reset_index()
    
    # 按recall排序
    data_df = data_df.sort_values('recall')
    
    return data_df['recall'].values, data_df['qps'].values

def load_result_data_dco(filename, dataset):
    df = pd.read_csv(filename)
    df_dataset = df[df['dataset'] == dataset]
    recall = df_dataset['recall'].values / 100
    Qps = 1e6 / (df_dataset['pruing_rate'].values * 1e3)
    # 创建DataFrame来处理重复的recall值
    data_df = pd.DataFrame({'recall': recall, 'qps': Qps})
    
    # 对于相同的recall值，取最小的qps值
    data_df = data_df.groupby('recall')['qps'].min().reset_index()
    
    # 按recall排序
    data_df = data_df.sort_values('recall')
    
    return data_df['recall'].values, data_df['qps'].values

def load_result_data_dade_dco(filename, dataset):
    df = pd.read_csv(filename)
    df_dataset = df[df['dataset'] == dataset]
    recall = df_dataset['recall'].values / 100
    if dataset == 'glove-200-angular' or dataset == 'sift-128-euclidean' or dataset == 'contriever-768' or dataset == 'deep-image-96-angular':
        Qps = 1e6 / (df_dataset['distance_time'].values * 1e2)
    else:
        Qps = 1e6 / (df_dataset['distance_time'].values * 1e3)
    # 创建DataFrame来处理重复的recall值
    data_df = pd.DataFrame({'recall': recall, 'qps': Qps})
    
    # 对于相同的recall值，取最小的qps值
    data_df = data_df.groupby('recall')['qps'].min().reset_index()
    
    # 按recall排序
    data_df = data_df.sort_values('recall')
    
    return data_df['recall'].values, data_df['qps'].values


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
                    label = "ivf-simd"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/ivf/simd/IVF_RES_simd_0_dist_time.csv"
                elif i == 1:
                    label = "ivf-adsampling-simd"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/ivf/simd/IVF_RES_simd_1_dist_time.csv"
                elif i == 2:
                    label = "ivf-bsa-learnedl2-simd"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/ivf/simd/IVF_RES_simd_3_dist_time.csv"
                elif i == 3:
                    label = "ivf-bsa-uniform-simd"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/ivf/simd/IVF_RES_simd_5_dist_time.csv"
                elif i == 4:
                    label = "ivf-dade-simd"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/DADEnew/IVF_DADE_3_SIMD_dist_time.csv"

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
        plt.savefig(f'E:/cppwork/dco_benchmarks/DATA/figure/RES/IVF/ivf_qps_subplots_@{K}_simd_dist_time.png', dpi=400, bbox_inches='tight')
        plt.show()

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
                    label = "ivf-nosimd"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/ivf/nosimd/IVF_RES_nosimd_0_dist_time.csv"
                elif i == 1:
                    label = "ivf-adsampling-simd"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/ivf/nosimd/IVF_RES_nosimd_1_dist_time.csv"
                elif i == 2:
                    label = "ivf-bsa-learnedl2-simd"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/ivf/nosimd/IVF_RES_nosimd_3_dist_time.csv"
                elif i == 3:
                    label = "ivf-bsa-uniform-simd"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/ivf/nosimd/IVF_RES_nosimd_5_dist_time.csv"
                elif i == 4:
                    label = "ivf-dade-nosimd"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/DADEnew/IVF_DADE_3_NOSIMD_dist_time.csv"

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
        plt.savefig(f'E:/cppwork/dco_benchmarks/DATA/figure/RES/IVF/ivf_qps_subplots_@{K}_nosimd_dist_time.png', dpi=400, bbox_inches='tight')
        plt.show()
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
                    label = "ivf-simd"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/ivf/simd/IVF_RES_simd_0_dist_time.csv"
                    recall, Qps = load_result_data_dco(filename, dataset)
                elif i == 1:
                    label = "ivf-adsampling-simd"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/ivf/simd/IVF_RES_simd_1_dist_time.csv"
                    recall, Qps = load_result_data_dco(filename, dataset)
                elif i == 2:
                    label = "ivf-bsa-learnedl2-simd"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/ivf/simd/IVF_RES_simd_3_dist_time.csv"
                    recall, Qps = load_result_data_dco(filename, dataset)
                elif i == 3:
                    label = "ivf-bsa-uniform-simd"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/ivf/simd/IVF_RES_simd_5_dist_time.csv"
                    recall, Qps = load_result_data_dco(filename, dataset)
                elif i == 4:
                    label = "ivf-dade-simd"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/DADEnew/IVF_DADE_3_SIMD_dist_time.csv"
                    recall, Qps = load_result_data_dade_dco(filename, dataset)

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
        plt.savefig(f'E:/cppwork/dco_benchmarks/DATA/figure/RES/IVF/ivf_dco_subplots_@{K}_simd_dist_time.png', dpi=400, bbox_inches='tight')
        plt.show()

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
                    label = "ivf-nosimd"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/ivf/nosimd/IVF_RES_nosimd_0_dist_time.csv"
                    recall, Qps = load_result_data_dco(filename, dataset)
                elif i == 1:
                    label = "ivf-adsampling-simd"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/ivf/nosimd/IVF_RES_nosimd_1_dist_time.csv"
                    recall, Qps = load_result_data_dco(filename, dataset)
                elif i == 2:
                    label = "ivf-bsa-learnedl2-simd"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/ivf/nosimd/IVF_RES_nosimd_3_dist_time.csv"
                    recall, Qps = load_result_data_dco(filename, dataset)
                elif i == 3:
                    label = "ivf-bsa-uniform-simd"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/ivf/nosimd/IVF_RES_nosimd_5_dist_time.csv"
                    recall, Qps = load_result_data_dco(filename, dataset)
                elif i == 4:
                    label = "ivf-dade-nosimd"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/DADEnew/IVF_DADE_3_NOSIMD_dist_time.csv"
                    recall, Qps = load_result_data_dade_dco(filename, dataset)

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
        plt.savefig(f'E:/cppwork/dco_benchmarks/DATA/figure/RES/IVF/ivf_dco_subplots_@{K}_nosimd_dist_time.png', dpi=400, bbox_inches='tight')
        plt.show()
