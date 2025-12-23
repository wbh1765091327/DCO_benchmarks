import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from utils import fvecs_read, fvecs_write
import os
import struct
from tqdm import tqdm
import pandas as pd
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
import matplotlib as mpl

# 基础字体
mpl.rcParams['font.family'] = 'Arial Unicode MS'

# 让 mathtext 使用 Arial Unicode MS
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Arial Unicode MS'
mpl.rcParams['mathtext.it'] = 'Arial Unicode MS'
mpl.rcParams['mathtext.bf'] = 'Arial Unicode MS'

datasets = ['glove-200-angular','deep-image-96-angular','contriever-768','instructorxl-arxiv-768','sift-128-euclidean','msong-420','gist-960-euclidean','openai-1536-angular']
datasets2 = ['GloVe','DEEP','Contriever','InstructorXL','SIFT','MSong','GIST','OpenAI']
ivf_marker = ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
col = ['#FF0000', '#0000FF', '#00AA00', '#FF8000', '#8000FF', '#FF1493', '#008B8B', '#B8860B', '#4B0082', '#228B22', '#FF4500']
# col = ['#E76254', '#EF8A47', '#F7AA58', '#FFDC6F', '#FFAAE7', '#AADED8', '#7290D5', '#52708D', '#376795', '#1E466E', '#FF4500']
# col = [
#     '#20364F',  # R:032 G:054 B:079
#     '#31646C',  # R:049 G:100 B:108
#     '#4E9296',  # R:078 G:146 B:128
#     '#96B89B',  # R:150 G:184 B:155
#     '#DCE1D2',  # R:220 G:223 B:210
#     '#ECD9CF',  # R:236 G:217 B:207
#     '#D49688',  # R:212 G:156 B:135
#     '#B85C65',  # R:184 G:098 B:101
#     '#8B3B5E',  # R:139 G:052 B:094
#     '#50304E',  # R:080 G:024 B:078
# ]

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

if __name__ == "__main__":
    # for K in [10]:
    #     n_rows = 2
    #     n_cols = (len(datasets) + 1) // 2
    #     fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 8), sharex=False, sharey=False)

    #     fig.subplots_adjust(hspace=0.3, wspace=0.2)
    #     axes = axes.flatten()

    #     for idx, dataset in enumerate(datasets):
    #         ax = axes[idx]
    #         for i in range(6):
    #             if i == 0:
    #                 label = "hnsw-simd"
    #                 filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/hnsw/simd/HNSW_RES_simd_0_dist_time.csv"
    #             elif i == 1:
    #                 label = "hnsw-adsampling-simd"
    #                 filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/hnsw/simd/HNSW_RES_simd_1_dist_time.csv"
    #             elif i == 2:
    #                 label = "hnsw-bsa-learnedl2-simd"
    #                 filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/hnsw/simd/HNSW_RES_simd_6_dist_time.csv"
    #             elif i == 3:
    #                 label = "hnsw-bsa-uniform-simd"
    #                 filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/hnsw/simd/HNSW_RES_simd_7_dist_time.csv"
    #             elif i == 4:
    #                 label = "hnsw-finger-simd"
    #                 filename = f"E:/cppwork/dco_benchmarks/DATA/Finger/HNSW_RES_simd_9_dist_time.csv"
    #             elif i == 5:
    #                 label = "hnsw-dade-simd"
    #                 filename = f"E:/cppwork/dco_benchmarks/DATA/DADEnew/HNSW_DADE_3_SIMD_dist_time.csv"

    #             recall, Qps = load_result_data(filename, dataset)
    #             mask = recall >= 0.80
    #             ax.plot(recall[mask], Qps[mask], marker=ivf_marker[i], c=col[i], label=label, alpha=0.5, linestyle="--", markerfacecolor='white')

    #         ax.set_title(dataset, fontsize=10)
    #         ax.grid(linestyle='--', linewidth=0.5)
    #         ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    #         ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    #         ax.set_ylim(bottom=0)
    #         ax.set_xlim(left=0.80)
    #         ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    #         ax.xaxis.set_major_locator(plt.MaxNLocator(4))

    #     # 设置公共x/y轴标签
    #     fig.text(0.5, 0.04, 'Recall@10', ha='center', fontsize=14)
    #     fig.text(0.06, 0.5, 'Qps', va='center', rotation='vertical', fontsize=14)

    #     handles, labels = ax.get_legend_handles_labels()
    #     fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=10)

    #     plt.rc('font', family='Arial Unicode MS')
    #     plt.savefig(f'E:/cppwork/dco_benchmarks/DATA/figure/RES/HNSW/hnsw_qps_subplots_@{K}_simd_dist_time.pdf', dpi=400, bbox_inches='tight')
    #     plt.show()

    for K in [10]:
        n_rows = 2
        n_cols = (len(datasets) + 1) // 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 7), sharex=False, sharey=False)

        fig.subplots_adjust(hspace=0.26, wspace=0.15, top=0.85, left=0.08, right=0.95, bottom=0.12)
        axes = axes.flatten()

        for idx, dataset in enumerate(datasets):
            ax = axes[idx]
            for i in range(9):
                if i == 0:
                    label = "HNSW"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/hnsw/nosimd/HNSW_RES_nosimd_0_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                elif i == 1:
                    label = "HNSW-ADS"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/hnsw/nosimd/HNSW_RES_nosimd_1_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                # elif i == 9:
                #     label = "HNSW-Finger"
                #     filename = f"E:/cppwork/dco_benchmarks/DATA/Finger/HNSW_RES_nosimd_9_dist_time.csv"
                elif i == 2:
                    label = "HNSW-DADE"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/DADEnew/HNSW_DADE_3_NOSIMD_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                elif i == 3:
                    label = r"HNSW-DDC$_{res}$"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/hnsw/nosimd/HNSW_RES_nosimd_7_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                elif i == 4:
                    label = r"HNSW-DDC$_{pca}$"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/hnsw/nosimd/HNSW_RES_nosimd_6_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                elif i == 5:
                    label = r"HNSW-DDC$_{opq}$"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Res-opq/hnsw/HNSW_RES_nosimd_3_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                elif i == 6:
                    label = "HNSW-RaBitQ"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/RabitQnew/HNSW_7BIT_NOSIMD_RabitQ_nosimd.csv"
                    recall, Qps = load_result_data_RabitQ(filename, dataset)
                elif i == 7 or i == 8:
                    continue

                mask = recall >= 0.80
                ax.plot(recall[mask], Qps[mask], marker=ivf_marker[i], c=col[i], label=label, alpha=0.5, linestyle="--", markerfacecolor='white', markersize=6, linewidth=2.5, markeredgecolor=col[i], markeredgewidth=1.5)

            # 使用简化的数据集名称作为标题
            ax.set_title(datasets2[idx], fontsize=18, fontfamily='Arial Unicode MS')
            ax.grid(linestyle='--', linewidth=0.5)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f'{y/1000:.1f}'))
            ax.set_ylim(bottom=0)
            ax.set_xlim(left=0.80)
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            # 增大坐标轴数字字体大小
            ax.tick_params(axis='both', which='major', labelsize=16, labelfontfamily='Arial Unicode MS')

        # 设置公共x/y轴标签
        fig.text(0.5, 0.04, 'Recall@10', ha='center', fontsize=18, fontfamily='Arial Unicode MS')
        fig.text(
            0.035, 0.5, r'$\mathrm{QPS}\ (\times 10^{3})$',
            va='center', rotation='vertical', fontsize=18
        )

        handles, labels = ax.get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=18, 
                          bbox_to_anchor=(0.5, 1.03), handlelength=3, handletextpad=1.2, columnspacing=1.5)
        legend.get_frame().set_edgecolor('none')  # 删除图例外边框
        legend.get_frame().set_facecolor('none')  # 删除图例背景

        plt.rc('font', family='Arial Unicode MS')
        plt.savefig(f'E:/cppwork/dco_benchmarks/DATA/figure/RES/HNSW/hnsw_all_nosimd_dist_time.pdf', dpi=400, bbox_inches='tight',format='pdf')
        plt.show()

    for K in [10]:
        n_rows = 2
        n_cols = (len(datasets) + 1) // 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 7), sharex=False, sharey=False)

        fig.subplots_adjust(hspace=0.26, wspace=0.15, top=0.85, left=0.08, right=0.95, bottom=0.12)
        axes = axes.flatten()

        for idx, dataset in enumerate(datasets):
            ax = axes[idx]
            for i in range(9):
                if i == 0:
                    label = "HNSW"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/hnsw/simd/HNSW_RES_simd_0_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                elif i == 1:
                    label = "HNSW-ADS"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/hnsw/simd/HNSW_RES_simd_1_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                # elif i == 9:
                #     label = "HNSW-Finger"
                #     filename = f"E:/cppwork/dco_benchmarks/DATA/Finger/HNSW_RES_simd_9_dist_time.csv"
                #     recall, Qps = load_result_data(filename, dataset)
                elif i == 2:
                    label = "HNSW-DADE"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/DADEnew/HNSW_DADE_3_SIMD_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                elif i == 3:
                    label = r"HNSW-DDC$_{res}$"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/hnsw/simd/HNSW_RES_simd_7_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                elif i == 4:
                    label = r"HNSW-DDC$_{pca}$"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/hnsw/simd/HNSW_RES_simd_6_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                elif i == 5:
                    label = r"HNSW-DDC$_{opq}$"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Res-opq-simd/hnsw/HNSW_RES_simd_4_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                elif i == 6:
                    label = "HNSW-RaBitQ"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/RabitQnew/HNSW_7BIT_RabitQ.csv"
                    recall, Qps = load_result_data_RabitQ(filename, dataset)
                elif i == 7 or i == 8:
                    continue
    
                mask = recall >= 0.80
                ax.plot(recall[mask], Qps[mask], marker=ivf_marker[i], c=col[i], label=label, alpha=0.5, linestyle="--", markerfacecolor='white', markersize=6, linewidth=2.5, markeredgecolor=col[i], markeredgewidth=1.5)

            ax.set_title(datasets2[idx], fontsize=18, fontfamily='Arial Unicode MS')
            ax.grid(linestyle='--', linewidth=0.5)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f'{y/1000:.1f}'))
            ax.set_ylim(bottom=0)
            ax.set_xlim(left=0.80)
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            # 增大坐标轴数字字体大小
            ax.tick_params(axis='both', which='major', labelsize=16, labelfontfamily='Arial Unicode MS')

        # 设置公共x/y轴标签
        fig.text(0.5, 0.04, 'Recall@10', ha='center', fontsize=18, fontfamily='Arial Unicode MS')
        fig.text(
            0.035, 0.5, r'$\mathrm{QPS}\ (\times 10^{3})$',
            va='center', rotation='vertical', fontsize=18
        )

        handles, labels = ax.get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=18, 
                          bbox_to_anchor=(0.5, 1.03), handlelength=3, handletextpad=1.2, columnspacing=1.5)
        legend.get_frame().set_edgecolor('none')  # 删除图例外边框
        legend.get_frame().set_facecolor('none')  # 删除图例背景

        plt.rc('font', family='Arial Unicode MS')
        plt.savefig(f'E:/cppwork/dco_benchmarks/DATA/figure/RES/HNSW/hnsw_all_simd_dist_time.pdf', dpi=400, bbox_inches='tight',format='pdf')
        plt.show()


    # for K in [10]:
    #     n_rows = 2
    #     n_cols = (len(datasets) + 1) // 2
    #     fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 8), sharex=False, sharey=False)

    #     fig.subplots_adjust(hspace=0.3, wspace=0.2)
    #     axes = axes.flatten()

    #     for idx, dataset in enumerate(datasets):
    #         ax = axes[idx]
    #         for i in range(6):
    #             if i == 0:
    #                 label = "hnsw-simd"
    #                 filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/hnsw/simd/HNSW_RES_simd_0_dist_time.csv"
    #                 recall, Qps = load_result_data_dco(filename, dataset)
    #             elif i == 1:
    #                 label = "hnsw-adsampling-simd"
    #                 filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/hnsw/simd/HNSW_RES_simd_1_dist_time.csv"
    #                 recall, Qps = load_result_data_dco(filename, dataset)
    #             elif i == 2:
    #                 label = "hnsw-bsa-learnedl2-simd"
    #                 filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/hnsw/simd/HNSW_RES_simd_6_dist_time.csv"
    #                 recall, Qps = load_result_data_dco(filename, dataset)
    #             elif i == 3:
    #                 label = "hnsw-bsa-uniform-simd"
    #                 filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/hnsw/simd/HNSW_RES_simd_7_dist_time.csv"
    #                 recall, Qps = load_result_data_dco(filename, dataset)
    #             elif i == 4:
    #                 label = "hnsw-finger-simd"
    #                 filename = f"E:/cppwork/dco_benchmarks/DATA/Finger/HNSW_RES_simd_9_dist_time.csv"
    #                 recall, Qps = load_result_data_dco(filename, dataset)
    #             elif i == 5:
    #                 label = "hnsw-dade"
    #                 filename = f"E:/cppwork/dco_benchmarks/DATA/DADEnew/HNSW_DADE_3_SIMD_dist_time.csv"
    #                 recall, Qps = load_result_data_dade_dco(filename, dataset)
                
    #             mask = recall >= 0.80
    #             ax.plot(recall[mask], Qps[mask], marker=ivf_marker[i], c=col[i], label=label, alpha=0.5, linestyle="--", markerfacecolor='white')

    #         ax.set_title(dataset, fontsize=10)
    #         ax.grid(linestyle='--', linewidth=0.5)
    #         ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    #         ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    #         ax.set_ylim(bottom=0)
    #         ax.set_xlim(left=0.80)
    #         ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    #         ax.xaxis.set_major_locator(plt.MaxNLocator(4))

    #     # 设置公共x/y轴标签
    #     fig.text(0.5, 0.04, 'Recall@10', ha='center', fontsize=14)
    #     fig.text(0.06, 0.5, 'Qps', va='center', rotation='vertical', fontsize=14)

    #     handles, labels = ax.get_legend_handles_labels()
    #     fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=10)

    #     plt.rc('font', family='Arial Unicode MS')
    #     plt.savefig(f'E:/cppwork/dco_benchmarks/DATA/figure/RES/HNSW/hnsw_dco_subplots_@{K}_simd_dist_time.pdf', dpi=400, bbox_inches='tight')
    #     plt.show()

    # for K in [10]:
    #     n_rows = 2
    #     n_cols = (len(datasets) + 1) // 2
    #     fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 8), sharex=False, sharey=False)

    #     fig.subplots_adjust(hspace=0.3, wspace=0.2)
    #     axes = axes.flatten()

    #     for idx, dataset in enumerate(datasets):
    #         ax = axes[idx]
    #         for i in range(6):
    #             if i == 0:
    #                 label = "hnsw"
    #                 filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/hnsw/nosimd/HNSW_RES_nosimd_0_dist_time.csv"
    #                 recall, Qps = load_result_data_dco(filename, dataset)
    #             elif i == 1:
    #                 label = "hnsw-adsampling"
    #                 filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/hnsw/nosimd/HNSW_RES_nosimd_1_dist_time.csv"
    #                 recall, Qps = load_result_data_dco(filename, dataset)
    #             elif i == 2:
    #                 label = "hnsw-bsa-learnedl2"
    #                 filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/hnsw/nosimd/HNSW_RES_nosimd_6_dist_time.csv"
    #                 recall, Qps = load_result_data_dco(filename, dataset)
    #             elif i == 3:
    #                 label = "hnsw-bsa-uniform"
    #                 filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/hnsw/nosimd/HNSW_RES_nosimd_7_dist_time.csv"
    #                 recall, Qps = load_result_data_dco(filename, dataset)
    #             elif i == 4:
    #                 label = "hnsw-finger"
    #                 filename = f"E:/cppwork/dco_benchmarks/DATA/Finger/HNSW_RES_nosimd_9_dist_time.csv"
    #                 recall, Qps = load_result_data_dco(filename, dataset)
    #             elif i == 5:
    #                 label = "hnsw-dade"
    #                 filename = f"E:/cppwork/dco_benchmarks/DATA/DADEnew/HNSW_DADE_3_NOSIMD_dist_time.csv"
    #                 recall, Qps = load_result_data_dade_dco(filename, dataset)

               
    #             mask = recall >= 0.80
    #             ax.plot(recall[mask], Qps[mask], marker=ivf_marker[i], c=col[i], label=label, alpha=0.5, linestyle="--", markerfacecolor='white')

    #         ax.set_title(dataset, fontsize=10)
    #         ax.grid(linestyle='--', linewidth=0.5)
    #         ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    #         ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    #         ax.set_ylim(bottom=0)
    #         ax.set_xlim(left=0.80)
    #         ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    #         ax.xaxis.set_major_locator(plt.MaxNLocator(4))

    #     # 设置公共x/y轴标签
    #     fig.text(0.5, 0.04, 'Recall@10', ha='center', fontsize=14)
    #     fig.text(0.06, 0.5, 'Qps', va='center', rotation='vertical', fontsize=14)

    #     handles, labels = ax.get_legend_handles_labels()
    #     fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=10)

    #     plt.rc('font', family='Arial Unicode MS')
    #     plt.savefig(f'E:/cppwork/dco_benchmarks/DATA/figure/RES/HNSW/hnsw_dco_subplots_@{K}_nosimd_dist_time.pdf', dpi=400, bbox_inches='tight')
    #     plt.show()