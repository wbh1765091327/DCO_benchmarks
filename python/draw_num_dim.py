import numpy as np
import matplotlib.pyplot as plt
from utils import fvecs_read, fvecs_write
import os
import struct
from tqdm import tqdm
import pandas as pd
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter

# datasets = ['glove-200-angular_1k','glove-200-angular_10k','glove-200-angular_100k','glove-200-angula_1000k']
# datasets2 = ['GloVe-200-1k','GloVe-200-10k','GloVe-200-100k','GloVe-200-1000k']
datasets = ['glove-25-angular_100k','glove-50-angular_100k','glove-100-angular_100k','glove-200-angular_100k','instructorxl-arxiv-768_1k','instructorxl-arxiv-768_10k','instructorxl-arxiv-768_100k','instructorxl-arxiv-768_1000k']
datasets2 = ['GloVe-25-100k','GloVe-50-100k','GloVe-100-100k','GloVe-200-100k','Instructorxl-768-1k','Instructorxl-768-10k','Instructorxl-768-100k','Instructorxl-768-1000k']
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

file_path_mapping = {
    'instructorxl-arxiv-768_1k': {
        'base': 'E:/cppwork/dco_benchmarks/DATA/Resnew/数据规模和维度-d32/recall@10/ivf/simd/',
        'dade': 'E:/cppwork/dco_benchmarks/DATA/DADEnew/数据规模分布-32/',
        'pdx': 'E:/cppwork/dco_benchmarks/DATA/PDXnew/规模大小i/',
        'rabitq': 'E:/cppwork/dco_benchmarks/DATA/RabitQnew/数据规模和分布新/'
    },
    'instructorxl-arxiv-768_10k': {
        'base': 'E:/cppwork/dco_benchmarks/DATA/Resnew/数据规模和维度-d32/recall@10/ivf/simd/',
        'dade': 'E:/cppwork/dco_benchmarks/DATA/DADEnew/数据规模分布-32/',
        'pdx': 'E:/cppwork/dco_benchmarks/DATA/PDXnew/规模大小i/',
        'rabitq': 'E:/cppwork/dco_benchmarks/DATA/RabitQnew/数据规模和分布新/'
    },
    'instructorxl-arxiv-768_100k': {
        'base': 'E:/cppwork/dco_benchmarks/DATA/Resnew/数据规模和维度-d32/recall@10/ivf/simd/',
        'dade': 'E:/cppwork/dco_benchmarks/DATA/DADEnew/数据规模分布-32/',
        'pdx': 'E:/cppwork/dco_benchmarks/DATA/PDXnew/规模大小i/',
        'rabitq': 'E:/cppwork/dco_benchmarks/DATA/RabitQnew/数据规模和分布新/'
    },
    'instructorxl-arxiv-768_1000k': {
        'base': 'E:/cppwork/dco_benchmarks/DATA/Resnew/数据规模和维度-d32/recall@10/ivf/simd/',
        'dade': 'E:/cppwork/dco_benchmarks/DATA/DADEnew/数据规模分布-32/',
        'pdx': 'E:/cppwork/dco_benchmarks/DATA/PDXnew/规模大小i/',
        'rabitq': 'E:/cppwork/dco_benchmarks/DATA/RabitQnew/数据规模和分布新/'
    },
    'glove-25-angular_100k': {
        'base': 'E:/cppwork/dco_benchmarks/DATA/Resnew/数据规模和维度-d16/ivf/simd/',
        'dade': 'E:/cppwork/dco_benchmarks/DATA/DADEnew/数据规模分布-16/',
        'pdx': 'E:/cppwork/dco_benchmarks/DATA/PDXnew/维度大小/',
        'rabitq': 'E:/cppwork/dco_benchmarks/DATA/RabitQnew/数据规模和分布/'
    },
    'glove-50-angular_100k': {
        'base': 'E:/cppwork/dco_benchmarks/DATA/Resnew/数据规模和维度-d16/ivf/simd/',
        'dade': 'E:/cppwork/dco_benchmarks/DATA/DADEnew/数据规模分布-16/',
        'pdx': 'E:/cppwork/dco_benchmarks/DATA/PDXnew/维度大小/',
        'rabitq': 'E:/cppwork/dco_benchmarks/DATA/RabitQnew/数据规模和分布/'
    },
    'glove-100-angular_100k': {
        'base': 'E:/cppwork/dco_benchmarks/DATA/Resnew/数据规模和维度-d16/ivf/simd/',
        'dade': 'E:/cppwork/dco_benchmarks/DATA/DADEnew/数据规模分布-16/',
        'pdx': 'E:/cppwork/dco_benchmarks/DATA/PDXnew/维度大小/',
        'rabitq': 'E:/cppwork/dco_benchmarks/DATA/RabitQnew/数据规模和分布/'
    },
    'glove-200-angular_100k': {
        'base': 'E:/cppwork/dco_benchmarks/DATA/Resnew/数据规模和维度-d16/ivf/simd/',
        'dade': 'E:/cppwork/dco_benchmarks/DATA/DADEnew/数据规模分布-16/',
        'pdx': 'E:/cppwork/dco_benchmarks/DATA/PDXnew/维度大小/',
        'rabitq': 'E:/cppwork/dco_benchmarks/DATA/RabitQnew/数据规模和分布/'
    },
}

if __name__ == "__main__":
    for K in [10]:
        n_rows = 2
        n_cols = 4
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 7), sharex=False, sharey=False)

        fig.subplots_adjust(hspace=0.26, wspace=0.15, top=0.85, left=0.08, right=0.95, bottom=0.12)
        axes = axes.flatten()

        for idx, dataset in enumerate(datasets):
            paths = file_path_mapping[dataset]
            ax = axes[idx]
            for i in range(10):
                if i == 0:
                    label = "IVF"
                    filename = f"{paths['base']}IVF_RES_simd_0_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                elif i == 1:
                    label = "IVF-ADS"
                    filename = f"{paths['base']}IVF_RES_simd_1_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                elif i == 2:
                    label = "IVF-DADE"
                    filename = f"{paths['dade']}IVF_DADE_3_SIMD_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                elif i == 3:
                    label = r"IVF-DDC$_{res}$"
                    filename = f"{paths['base']}IVF_RES_simd_5_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                elif i == 4:
                    label = r"IVF-DDC$_{pca}$"
                    filename = f"{paths['base']}IVF_RES_simd_3_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                elif i == 5:
                    label = r"IVF-DDC$_{opq}$"
                    filename = f"{paths['base']}IVF_RES_simd_6_dist_time.csv"
                    recall, Qps = load_result_data(filename, dataset)
                elif i == 8:
                    label = "IVF-PDX-ADS"
                    filename = f"{paths['pdx']}IVF_PDX_ADSAMPLING_simd.csv"
                    recall, Qps = load_result_data_pdx(filename, dataset)
                elif i == 9:
                    label = "IVF-PDX-BOND"
                    filename = f"{paths['pdx']}IVF_PDX_BOND_simd.csv"
                    recall, Qps = load_result_data_pdx(filename, dataset)
                elif i == 6:
                    label = "IVF-RaBitQ"
                    filename = f"{paths['rabitq']}IVF_7BIT_RabitQ.csv"
                    recall, Qps = load_result_data_RabitQ(filename, dataset)
                elif i == 7:
                    continue

                mask = recall >= 0.80
                ax.plot(recall[mask], Qps[mask], marker=ivf_marker[i], c=col[i], label=label, alpha=0.5, linestyle="--", markerfacecolor='white', markersize=6, linewidth=2.5, markeredgecolor=col[i], markeredgewidth=1.5)


            ax.set_title(datasets2[idx], fontsize=21, fontfamily='Times New Roman')
            ax.grid(linestyle='--', linewidth=0.5)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f'{y/1000:.1f}'))
            ax.set_ylim(bottom=0)
            ax.set_xlim(left=0.80)
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            # 增大坐标轴数字字体大小
            ax.tick_params(axis='both', which='major', labelsize=16, labelfontfamily='Times New Roman')

        # 设置公共x/y轴标签
        fig.text(0.5, -0.02, 'Recall@10', ha='center', fontsize=21, fontfamily='Times New Roman')
        fig.text(0.035, 0.5, 'QPS×10$^3$', va='center', rotation='vertical', fontsize=21, fontfamily='Times New Roman')

        handles, labels = ax.get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=19, 
                          bbox_to_anchor=(0.5, 1.16), handlelength=3, handletextpad=1.2, columnspacing=1)
        legend.get_frame().set_edgecolor('none')  # 删除图例外边框
        legend.get_frame().set_facecolor('none')  # 删除图例背景

        plt.rc('font', family='Times New Roman')
        plt.savefig(f'E:/cppwork/dco_benchmarks/DATA/figure/ALL/IVF_all_simd_dim_num.png', dpi=400, bbox_inches='tight',format='png')
        plt.show()