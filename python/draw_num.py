import numpy as np
import matplotlib.pyplot as plt
from utils import fvecs_read, fvecs_write
import os
import struct
from tqdm import tqdm
import pandas as pd
import matplotlib.ticker as ticker

# datasets = ['glove-200-angular_1k','glove-200-angular_10k','glove-200-angular_100k','glove-200-angula_1000k']
# datasets2 = ['GloVe-200-1k','GloVe-200-10k','GloVe-200-100k','GloVe-200-1000k']
datasets = ['instructorxl-arxiv-768_1k','instructorxl-arxiv-768_10k','instructorxl-arxiv-768_100k','instructorxl-arxiv-768_1000k']
datasets2 = ['Instructorxl-768-1k','Instructorxl-768-10k','Instructorxl-768-100k','Instructorxl-768-1000k']
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



if __name__ == "__main__":
    for K in [10]:
        n_rows = 2
        n_cols = (len(datasets) + 1) // 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 8), sharex=False, sharey=False)

        fig.subplots_adjust(hspace=0.3, wspace=0.2, top=0.85, left=0.08, right=1, bottom=0.12)
        axes = axes.flatten()

        for idx, dataset in enumerate(datasets):
            ax = axes[idx]
            for i in range(10):
                if i == 0:
                    label = "IVF"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/数据规模和维度-d32/recall@10/ivf/nosimd/IVF_RES_nosimd_0_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                elif i == 1:
                    label = "IVF-ADS"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/数据规模和维度-d32/recall@10/ivf/nosimd/IVF_RES_nosimd_1_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                elif i == 2:
                    label = "IVF-DADE"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/DADEnew/数据规模分布-32/IVF_DADE_3_NOSIMD_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                elif i == 3:
                    label = r"IVF-DDC$_{res}$"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/数据规模和维度-d32/recall@10/ivf/nosimd/IVF_RES_nosimd_5_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                elif i == 4:
                    label = r"IVF-DDC$_{pca}$"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/数据规模和维度-d32/recall@10/ivf/nosimd/IVF_RES_nosimd_3_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                elif i == 5:
                    label = r"IVF-DDC$_{opq}$"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/数据规模和维度-d32/recall@10/ivf/nosimd/IVF_RES_nosimd_6_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                elif i == 8:
                    label = "IVF-PDX-ADS"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/PDXnew/规模大小i/IVF_PDX_ADSAMPLING_nosimd.csv"
                    recall, Qps = load_result_data_pdx(filename, dataset)
                elif i == 9:
                    label = "IVF-PDX-BOND"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/PDXnew/规模大小i/IVF_PDX_BOND_nosimd.csv"
                    recall, Qps = load_result_data_pdx(filename, dataset)
                elif i == 6:
                    continue
                elif i == 7:
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
        fig.text(0.53, 0.04, 'Recall@10', ha='center', fontsize=22)
        fig.text(0, 0.47, 'Qps', va='center', rotation='vertical', fontsize=22)

        handles, labels = ax.get_legend_handles_labels()
        legend = fig.legend(handles, labels, 
           loc='center left', # 定位在左侧中心
           bbox_to_anchor=(1, 0.5), # 将其锚定在整个图表的右边界外
           ncol=1, # 单列显示
           fontsize=18)
        legend.get_frame().set_edgecolor('none')  # 删除图例外边框
        legend.get_frame().set_facecolor('none')  # 删除图例背景


        # plt.rc('font', family='Times New Roman')
        plt.savefig(f'E:/cppwork/dco_benchmarks/DATA/figure/数据规模新/IVF_all_nosimd_num.png', dpi=400, bbox_inches='tight',format='png')
        plt.show()


    for K in [10]:
        n_rows = 2
        n_cols = (len(datasets) + 1) // 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 8), sharex=False, sharey=False)

        fig.subplots_adjust(hspace=0.3, wspace=0.2, top=0.85, left=0.08, right=1, bottom=0.12)
        axes = axes.flatten()

        for idx, dataset in enumerate(datasets):
            ax = axes[idx]
            for i in range(10):
                if i == 0:
                    label = "IVF"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/数据规模和维度-d32/recall@10/ivf/simd/IVF_RES_simd_0_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                elif i == 1:
                    label = "IVF-ADS"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/数据规模和维度-d16/ivf/simd/IVF_RES_simd_1_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                elif i == 2:
                    label = "IVF-DADE"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/DADEnew/数据规模分布-32/IVF_DADE_3_SIMD_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                elif i == 3:
                    label = r"IVF-DDC$_{res}$"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/数据规模和维度-d32/recall@10/ivf/simd/IVF_RES_simd_5_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                elif i == 4:
                    label = r"IVF-DDC$_{pca}$"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/数据规模和维度-d32/recall@10/ivf/simd/IVF_RES_simd_3_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                elif i == 5:
                    label = r"IVF-DDC$_{opq}$"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/数据规模和维度-d32/recall@10/ivf/simd/IVF_RES_simd_6_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                elif i == 8:
                    label = "IVF-PDX-ADS"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/PDXnew/规模大小i/IVF_PDX_ADSAMPLING_simd.csv"
                    recall, Qps = load_result_data_pdx(filename, dataset)
                elif i == 9:
                    label = "IVF-PDX-BOND"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/PDXnew/规模大小i/IVF_PDX_BOND_simd.csv"
                    recall, Qps = load_result_data_pdx(filename, dataset)
                elif i == 6:
                    label = "IVF-RaBitQ"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/RabitQnew/数据规模和分布新/IVF_7BIT_RabitQ.csv"
                    recall, Qps = load_result_data_RabitQ(filename, dataset)
                elif i == 7:
                    continue

                mask = recall >= 0.80
                ax.plot(recall[mask], Qps[mask], marker=ivf_marker[i], c=col[i], label=label, alpha=0.5, linestyle="--", markerfacecolor='white', markersize=6, linewidth=2.5, markeredgecolor=col[i], markeredgewidth=1.5)

            ax.set_title(datasets2[idx], fontsize=22, fontfamily='Times New Roman')
            # ax.grid(linestyle='--', linewidth=0.5)
            ax.grid(True, which="major", linestyle="--", linewidth=0.5)
            ax.minorticks_on()
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
            ax.set_yscale('log')
            ax.set_ylim(bottom=Qps.min(),top=41130)
            ax.set_yticks([1e5]) 
            ax.set_xlim(left=0.80)
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            ax.tick_params(axis='both', which='major', labelsize=16)

        # 设置公共x/y轴标签
        fig.text(0.55, 0.04, 'Recall@10', ha='center', fontsize=22)
        fig.text(0, 0.5, 'Qps', va='center', rotation='vertical', fontsize=22)

        handles, labels = ax.get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=22, 
                          bbox_to_anchor=(0.5, 1.1), handlelength=2.5, handletextpad=1.2, columnspacing=1.5)
        legend.get_frame().set_edgecolor('none')  # 删除图例外边框
        legend.get_frame().set_facecolor('none')  # 删除图例背景

        # plt.rc('font', family='Times New Roman')
        plt.savefig(f'E:/cppwork/dco_benchmarks/DATA/figure/数据规模新/IVF_all_simd_num.png', dpi=400, bbox_inches='tight',format='png')
        plt.show()
    
    for K in [10]:
        n_rows = 2
        n_cols = (len(datasets) + 1) // 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 8), sharex=False, sharey=False)

        fig.subplots_adjust(hspace=0.3, wspace=0.2, top=0.85, left=0.08, right=1, bottom=0.12)
        axes = axes.flatten()

        for idx, dataset in enumerate(datasets):
            ax = axes[idx]
            for i in range(9):
                if i == 0:
                    label = "HNSW"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/数据规模和维度-d32/recall@10/hnsw/nosimd/HNSW_RES_nosimd_0_dist_time.csv"
                elif i == 1:
                    label = "HNSW-ADS"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/数据规模和维度-d32/recall@10/hnsw/nosimd/HNSW_RES_nosimd_1_dist_time.csv"
                elif i == 2:
                    label = "HNSW-DADE"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/DADEnew/数据规模分布-32/HNSW_DADE_3_NOSIMD_dist_time.csv"
                elif i == 3:
                    label = r"HNSW-DDC$_{res}$"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/数据规模和维度-d32/recall@10/hnsw/nosimd/HNSW_RES_nosimd_7_dist_time.csv"
                elif i == 4:
                    label = r"HNSW-DDC$_{pca}$"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/数据规模和维度-d32/recall@10/hnsw/nosimd/HNSW_RES_nosimd_6_dist_time.csv"
                elif i == 5:
                    label = r"HNSW-DDC$_{opq}$"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/数据规模和维度-d32/recall@10/hnsw/nosimd/HNSW_RES_nosimd_3_dist_time.csv"
                elif i == 6 or i == 7 or i == 8:
                    continue

                recall, Qps = load_result_data(filename, dataset)
                mask = recall >= 0.80
                ax.plot(recall[mask], Qps[mask], marker=ivf_marker[i], c=col[i], label=label, alpha=0.5, linestyle="--", markerfacecolor='white', markersize=6, linewidth=2.5, markeredgecolor=col[i], markeredgewidth=1.5)

            # 使用简化的数据集名称作为标题
            ax.set_title(datasets2[idx], fontsize=22, fontfamily='Times New Roman')
            ax.grid(linestyle='--', linewidth=0.5)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
            ax.set_ylim(bottom=0)
            ax.set_xlim(left=0.80)
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            # 增大坐标轴数字字体大小
            ax.tick_params(axis='both', which='major', labelsize=16)

        # 设置公共x/y轴标签
        fig.text(0.53, 0.04, 'Recall@10', ha='center', fontsize=22)
        fig.text(0, 0.47, 'Qps', va='center', rotation='vertical', fontsize=22)

        handles, labels = ax.get_legend_handles_labels()
        legend = fig.legend(handles, labels, 
           loc='center left', # 定位在左侧中心
           bbox_to_anchor=(1, 0.5), # 将其锚定在整个图表的右边界外
           ncol=1, # 单列显示
           fontsize=18)
        legend.get_frame().set_edgecolor('none')  # 删除图例外边框
        legend.get_frame().set_facecolor('none')  # 删除图例背景

        # plt.rc('font', family='Times New Roman')
        plt.savefig(f'E:/cppwork/dco_benchmarks/DATA/figure/数据规模新/hnsw_all_nosimd_num.png', dpi=400, bbox_inches='tight',format='png')
        plt.show()

    for K in [10]:
        n_rows = 2
        n_cols = (len(datasets) + 1) // 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 8), sharex=False, sharey=False)

        fig.subplots_adjust(hspace=0.3, wspace=0.2, top=0.85, left=0.08, right=1, bottom=0.12)
        axes = axes.flatten()

        for idx, dataset in enumerate(datasets):
            ax = axes[idx]
            for i in range(9):
                if i == 0:
                    label = "HNSW"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/数据规模和维度-d32/recall@10/hnsw/simd/HNSW_RES_simd_0_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                elif i == 1:
                    label = "HNSW-ADS"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/数据规模和维度-d32/recall@10/hnsw/simd/HNSW_RES_simd_1_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                elif i == 2:
                    label = "HNSW-DADE"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/DADEnew/数据规模分布-32/HNSW_DADE_3_SIMD_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                elif i == 3:
                    label = r"HNSW-DDC$_{res}$"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/数据规模和维度-d32/recall@10/hnsw/simd/HNSW_RES_simd_7_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                elif i == 4:
                    label = r"HNSW-DDC$_{pca}$"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/数据规模和维度-d32/recall@10/hnsw/simd/HNSW_RES_simd_6_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                elif i == 5:
                    label = r"HNSW-DDC$_{opq}$"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/数据规模和维度-d32/recall@10/hnsw/simd/HNSW_RES_simd_4_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                elif i == 6:
                    label = "HNSW-RaBitQ"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/RabitQnew/数据规模和分布新/HNSW_7BIT_RabitQ.csv"
                    recall, Qps = load_result_data_RabitQ(filename, dataset)
                elif i == 7 or i == 8:
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
            # 增大坐标轴数字字体大小
            ax.tick_params(axis='both', which='major', labelsize=16)

        # 设置公共x/y轴标签
        fig.text(0.53, 0.04, 'Recall@10', ha='center', fontsize=22)
        fig.text(0, 0.47, 'Qps', va='center', rotation='vertical', fontsize=22)

        handles, labels = ax.get_legend_handles_labels()
        legend = fig.legend(handles, labels, 
           loc='center left', # 定位在左侧中心
           bbox_to_anchor=(1, 0.5), # 将其锚定在整个图表的右边界外
           ncol=1, # 单列显示
           fontsize=18)
        legend.get_frame().set_edgecolor('none')  # 删除图例外边框
        legend.get_frame().set_facecolor('none')  # 删除图例背景


        # plt.rc('font', family='Times New Roman')
        plt.savefig(f'E:/cppwork/dco_benchmarks/DATA/figure/数据规模新/hnsw_all_simd_num.png', dpi=400, bbox_inches='tight',format='png')
        plt.show()