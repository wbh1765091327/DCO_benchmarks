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
col = ['#FF0000', '#0000FF', '#00AA00', '#FF8000', '#8000FF', '#FF1493', '#008B8B', '#B8860B', '#4B0082', '#DC143C', '#228B22', '#FF4500']

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

def load_result_data_RabitQ(filename, dataset):
    df = pd.read_csv(filename)
    df_dataset = df[df['dataset'] == dataset]
    recall = df_dataset['per_query_us'].values 
    Qps = df_dataset['recall'].values
    return recall, Qps

def load_result_data_pdx(filename, dataset):
    df = pd.read_csv(filename)
    # Filter data for specific dataset
    df_dataset = df[df['dataset'] == dataset]
    # Get recall and QPS data
    recall = df_dataset['recall'].values
    # Calculate QPS (queries per second)
    Qps = 1e3 / df_dataset['avg_all'].values  # Convert average query time (ms) to QPS
    return recall, Qps

def load_result_data_pdx_dco(filename, dataset):
    df = pd.read_csv(filename)
    # Filter data for specific dataset
    df_dataset = df[df['dataset'] == dataset]
    # Get recall and QPS data
    recall = df_dataset['recall'].values
    # Calculate QPS (queries per second)
    Qps = 1e3 / df_dataset['avg_distance_calculation'].values  # Convert average query time (ms) to QPS
    return recall, Qps

def load_result_data_Tribase(filename, dataset):
    df = pd.read_csv(filename)
    df_dataset = df[df['dataset'] == dataset]
    if dataset == 'msong-420' or dataset == 'gist-960-euclidean' or dataset == 'openai-1536-angular' or dataset == 'instructorxl-arxiv-768':
        df.query = df_dataset['query_time'].values
        recall = df_dataset['recall'].values
        Qps = 1e3 / df_dataset['query_time'].values 
    else:
        df.query = df_dataset['query_time'].values
        recall = df_dataset['recall'].values
        Qps = 1e4 / df_dataset['query_time'].values 
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
            for i in range(4):
                if i == 0:
                    label = "ivf-simd"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/PDX/IVF_FAISS.csv"
                    recall, Qps = load_result_data_pdx(filename, dataset)
                elif i == 1:
                    label = "ivf-adsampling-simd"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/ivf/simd/IVF_RES_simd_1_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                # elif i == 2:
                #     label = "ivf-bsa-learnedl2-simd"
                #     filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/ivf/simd/IVF_RES_simd_3_dist_time.csv"
                #     recall, Qps = load_result_data(filename, dataset)
                # elif i == 3:
                #     label = "ivf-bsa-uniform-simd"
                #     filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/ivf/simd/IVF_RES_simd_5_dist_time.csv"
                #     recall, Qps = load_result_data(filename, dataset)
                elif i == 2:
                    label = "ivf-adsmapling-pdx"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/PDXnew/split/pdx_adsampling.csv"
                    recall, Qps = load_result_data_pdx(filename, dataset)
                # elif i == 5:
                #     label = "ivf-bsa-uniform-pdx"
                #     filename = f"E:/cppwork/dco_benchmarks/DATA/PDX/IVF_PDX_BSA.csv"
                #     recall, Qps = load_result_data_pdx(filename, dataset)
                elif i == 3:
                    label = "ivf-bond-pdx"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/PDXnew/split/pdx_bond_sec.csv"
                    recall, Qps = load_result_data_pdx(filename, dataset)

                
                mask = recall >= 0.80
                ax.plot(recall[mask], Qps[mask], marker=ivf_marker[i], c=col[i], label=label, alpha=0.5, linestyle="--", markerfacecolor='white', markersize=6, linewidth=2.5, markeredgecolor=col[i], markeredgewidth=1.5)

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
        plt.savefig(f'E:/cppwork/dco_benchmarks/DATA/figure/PDX-ALL/IVF/pdx-ads-bond.png', dpi=400, bbox_inches='tight')
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
                if i == 0:
                    label = "ivf-simd"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/PDX/IVF_FAISS.csv"
                    recall, Qps = load_result_data_pdx(filename, dataset)
                elif i == 1:
                    label = "ivf-adsampling-simd"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/ivf/simd/IVF_RES_simd_1_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                # elif i == 2:
                #     label = "ivf-bsa-learnedl2-simd"
                #     filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/ivf/simd/IVF_RES_simd_3_dist_time.csv"
                #     recall, Qps = load_result_data(filename, dataset)
                # elif i == 3:
                #     label = "ivf-bsa-uniform-simd"
                #     filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/ivf/simd/IVF_RES_simd_5_dist_time.csv"
                #     recall, Qps = load_result_data(filename, dataset)
                elif i == 2:
                    label = "ivf-adsmapling-pdx"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/PDXnew/split/pdx_adsampling.csv"
                    recall, Qps = load_result_data_pdx_dco(filename, dataset)
                # elif i == 5:
                #     label = "ivf-bsa-uniform-pdx"
                #     filename = f"E:/cppwork/dco_benchmarks/DATA/PDX/IVF_PDX_BSA.csv"
                #     recall, Qps = load_result_data_pdx(filename, dataset)
                elif i == 3:
                    label = "ivf-bond-pdx"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/PDXnew/split/pdx_bond_sec.csv"
                    recall, Qps = load_result_data_pdx_dco(filename, dataset)

                
                mask = recall >= 0.80
                ax.plot(recall[mask], Qps[mask], marker=ivf_marker[i], c=col[i], label=label, alpha=0.5, linestyle="--", markerfacecolor='white', markersize=6, linewidth=2.5, markeredgecolor=col[i], markeredgewidth=1.5)

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
        plt.savefig(f'E:/cppwork/dco_benchmarks/DATA/figure/PDX-ALL/IVF/pdx-ads-bond-dco.png', dpi=400, bbox_inches='tight')
        plt.show()  

   
