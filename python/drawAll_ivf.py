import numpy as np
import matplotlib.pyplot as plt
from utils import fvecs_read, fvecs_write
import os
import struct
from tqdm import tqdm
import pandas as pd
import matplotlib.ticker as ticker

datasets = ['glove-200-angular','deep-image-96-angular','contriever-768','instructorxl-arxiv-768','sift-128-euclidean','msong-420','gist-960-euclidean','openai-1536-angular']
datasets2 = ['GloVe','DEEP','Contriever','Instructorxl','SIFT','MSong','GIST','OpenAI']
ivf_marker = ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
col = ['#FF0000', '#0000FF', '#00AA00', '#FF8000', '#8000FF', '#FF1493', '#008B8B', '#B8860B', '#4B0082', '#228B22', '#FF4500']

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

def load_result_data_RabitQ(filename, dataset):
    df = pd.read_csv(filename)
    df_dataset = df[df['dataset'] == dataset]
    recall = df_dataset['per_query_us'].values 
    Qps = df_dataset['recall'].values
    # 创建DataFrame来处理重复的recall值
    data_df = pd.DataFrame({'recall': recall, 'qps': Qps})
    
    # 对于相同的recall值，取最小的qps值
    data_df = data_df.groupby('recall')['qps'].min().reset_index()
    
    # 按recall排序
    data_df = data_df.sort_values('recall')
    
    return data_df['recall'].values, data_df['qps'].values

def load_result_data_pdx(filename, dataset):
    df = pd.read_csv(filename)
    # Filter data for specific dataset
    df_dataset = df[df['dataset'] == dataset]
    # Get recall and QPS data
    recall = df_dataset['recall'].values
    # Calculate QPS (queries per second)
    Qps = 1e3 / df_dataset['avg_all'].values  # Convert average query time (ms) to QPS
    # 创建DataFrame来处理重复的recall值
    data_df = pd.DataFrame({'recall': recall, 'qps': Qps})
    
    # 对于相同的recall值，取最小的qps值
    data_df = data_df.groupby('recall')['qps'].min().reset_index()
    
    # 按recall排序
    data_df = data_df.sort_values('recall')
    
    return data_df['recall'].values, data_df['qps'].values

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

        fig.subplots_adjust(hspace=0.3, wspace=0.2, top=0.85, left=0.08, right=0.95, bottom=0.12)
        axes = axes.flatten()

        for idx, dataset in enumerate(datasets):
            ax = axes[idx]
            for i in range(10):
                if i == 0:
                    label = "IVF"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/ivf/nosimd/IVF_RES_nosimd_0_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                elif i == 1:
                    label = "IVF-ADS"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/ivf/nosimd/IVF_RES_nosimd_1_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                elif i == 2:
                    label = "IVF-DADE"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/DADEnew/IVF_DADE_3_NOSIMD_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                elif i == 3:
                    label = r"IVF-DDC$_{res}$"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/ivf/nosimd/IVF_RES_nosimd_5_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                elif i == 4:
                    label = r"IVF-DDC$_{pca}$"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/ivf/nosimd/IVF_RES_nosimd_3_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                elif i == 5:
                    label = r"IVF-DDC$_{opq}$"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Res-opq/ivf/IVF_RES_nosimd_6_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                elif i == 7:
                    label = "IVF-Tribase"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Tribase/IVF_Tribase_nosimd_new.csv"
                    recall, Qps = load_result_data_Tribase(filename, dataset)
                elif i == 8:
                    label = "IVF-PDX-ADS"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/PDXnew/IVF_PDX_ADSAMPLING_nosimd.csv"
                    recall, Qps = load_result_data_pdx(filename, dataset)
                elif i == 9:
                    label = "IVF-PDX-BOND"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/PDXnew/IVF_PDX_BOND_nosimd.csv"
                    recall, Qps = load_result_data_pdx(filename, dataset)
                elif i == 6:
                    continue

                mask = recall >= 0.80      
                ax.plot(recall[mask], Qps[mask], marker=ivf_marker[i], c=col[i], label=label, alpha=0.5, linestyle="--", markerfacecolor='white', markersize=6, linewidth=2.5, markeredgecolor=col[i], markeredgewidth=1.5)
            

            ax.set_title(datasets2[idx], fontsize=22, fontfamily='Times New Roman')
            ax.grid(linestyle='--', linewidth=0.5)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
            ax.set_ylim(bottom=0)
            ax.set_xlim(left=0.80)
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            ax.tick_params(axis='both', which='major', labelsize=16)

        # 设置公共x/y轴标签
        fig.text(0.5, 0.04, 'Recall@10', ha='center', fontsize=22)
        fig.text(0.04, 0.5, 'Qps', va='center', rotation='vertical', fontsize=22)

        handles, labels = ax.get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=22, 
                          bbox_to_anchor=(0.5, 1.03), handlelength=2.5, handletextpad=1.2, columnspacing=1.5)
        legend.get_frame().set_edgecolor('none')  # 删除图例外边框
        legend.get_frame().set_facecolor('none')  # 删除图例背景


        # plt.rc('font', family='Times New Roman')
        plt.savefig(f'E:/cppwork/dco_benchmarks/DATA/figure/RES/IVF/IVF_all_nosimd_dist_time.pdf', dpi=400, bbox_inches='tight',format='pdf')
        plt.show()


    for K in [10]:
        n_rows = 2
        n_cols = (len(datasets) + 1) // 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 8), sharex=False, sharey=False)

        fig.subplots_adjust(hspace=0.3, wspace=0.2, top=0.85, left=0.08, right=0.95, bottom=0.12)
        axes = axes.flatten()

        for idx, dataset in enumerate(datasets):
            ax = axes[idx]
            for i in range(10):
                if i == 0:
                    label = "IVF"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/ivf/simd/IVF_RES_simd_0_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                elif i == 1:
                    label = "IVF-ADS"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/ivf/simd/IVF_RES_simd_1_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                elif i == 2:
                    label = "IVF-DADE"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/DADEnew/IVF_DADE_3_SIMD_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                elif i == 3:
                    label = r"IVF-DDC$_{res}$"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/ivf/simd/IVF_RES_simd_5_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                elif i == 4:
                    label = r"IVF-DDC$_{pca}$"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/ivf/simd/IVF_RES_simd_3_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                elif i == 5:
                    label = r"IVF-DDC$_{opq}$"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Res-opq-simd/ivf/IVF_RES_simd_6_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                elif i == 8:
                    label = "IVF-PDX-ADS"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/PDXnew/split/pdx_adsampling.csv"
                    recall, Qps = load_result_data_pdx(filename, dataset)
                elif i == 9:
                    label = "IVF-PDX-BOND"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/PDX/pdx-bond.csv"
                    recall, Qps = load_result_data_pdx(filename, dataset)
                elif i == 6:
                    label = "IVF-RaBitQ"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/RabitQnew/IVF_7BIT_RabitQ.csv"
                    recall, Qps = load_result_data_RabitQ(filename, dataset)
                elif i == 7:
                    label = "IVF-Tribase"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Tribase/IVF_Tribase_simd_new.csv"
                    recall, Qps = load_result_data_Tribase(filename, dataset)

                mask = recall >= 0.80
                ax.plot(recall[mask], Qps[mask], marker=ivf_marker[i], c=col[i], label=label, alpha=0.5, linestyle="--", markerfacecolor='white', markersize=6, linewidth=2.5, markeredgecolor=col[i], markeredgewidth=1.5)

            ax.set_title(datasets2[idx], fontsize=22, fontfamily='Times New Roman')
            ax.grid(linestyle='--', linewidth=0.5)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
            ax.set_ylim(bottom=0)
            ax.set_xlim(left=0.80)
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            ax.tick_params(axis='both', which='major', labelsize=16)

        # 设置公共x/y轴标签
        fig.text(0.5, 0.04, 'Recall@10', ha='center', fontsize=22)
        fig.text(0.04, 0.5, 'Qps', va='center', rotation='vertical', fontsize=22)

        handles, labels = ax.get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=22, 
                          bbox_to_anchor=(0.5, 1.03), handlelength=2.5, handletextpad=1.2, columnspacing=1.5)
        legend.get_frame().set_edgecolor('none')  # 删除图例外边框
        legend.get_frame().set_facecolor('none')  # 删除图例背景

        # plt.rc('font', family='Times New Roman')
        plt.savefig(f'E:/cppwork/dco_benchmarks/DATA/figure/RES/IVF/IVF_all_simd_dist_time.pdf', dpi=400, bbox_inches='tight',format='pdf')
        plt.show()
