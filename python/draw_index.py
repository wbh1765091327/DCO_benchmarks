import numpy as np
import matplotlib.pyplot as plt
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

ivf_marker = ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
col = ['#FF0000', '#0000FF', '#00AA00', '#FF8000', '#8000FF', '#FF1493', '#008B8B', '#B8860B', '#4B0082', '#228B22', '#FF4500']
Clist = [158,632]
Mlist = [8,16,32]
EFlist = [250,500,750]
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
def load_log(filepath):
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # 跳过空行
                    parts = line.split()
                    if len(parts) >= 3:  # 确保至少有5列数据
                        efsearch = int(parts[0])
                        recall = float(parts[1]) / 100  # 转换为0-1范围
                        per_query_us = float(parts[2])  # 使用微秒
                        
                        # 计算QPS (每秒查询数)
                        qps = 1e6 / per_query_us
                        
                        # 根据需要的列创建数据行
                        row = {}
                        row['efsearch'] = efsearch
                        row['recall'] = recall
                        row['per_query_us'] = per_query_us
                        row['qps'] = qps
                        data.append(row)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {filepath}")
        return pd.DataFrame()
    except Exception as e:
        print(f"读取文件 {filepath} 时出错: {e}")
        return pd.DataFrame()
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    if not df.empty:
        df = df.groupby('recall').agg({
            'efsearch': 'first',  # 保留第一个efsearch值
            'per_query_us': 'min',  # 取最小查询时间
            'qps': 'max'  # 取最大QPS（对应最小查询时间）
        }).reset_index()
        df = df.sort_values('recall')
    return df

def load_log_simple(filepath):
    """
    从日志文件中加载数据并返回recall和qps数组
    对于相同的recall值，取最大的qps值（最小查询时间对应最佳性能）
    """
    df = load_log(filepath)
    
    if df.empty:
        return np.array([]), np.array([])
    
    return df['recall'].values, df['qps'].values

def load_log_RaBitQ(filepath):
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # 跳过空行
                    parts = line.split()
                    if len(parts) >= 3:  # 确保至少有5列数据
                        efsearch = int(parts[0])
                        recall = float(parts[2]) # 转换为0-1范围
                        qps = float(parts[1])  # 使用微秒
            
                        
                        # 根据需要的列创建数据行
                        row = {}
                        row['efsearch'] = efsearch
                        row['recall'] = recall
                        row['qps'] = qps
                        data.append(row)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {filepath}")
        return pd.DataFrame()
    except Exception as e:
        print(f"读取文件 {filepath} 时出错: {e}")
        return pd.DataFrame()
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    if not df.empty:
        df = df.groupby('recall').agg({
            'efsearch': 'first',  # 保留第一个efsearch值
            'qps': 'max'  # 取最大QPS（对应最小查询时间）
        }).reset_index()
        df = df.sort_values('recall')
    return df

def load_log_simple_RaBitQ(filepath):
    """
    从日志文件中加载数据并返回recall和qps数组
    对于相同的recall值，取最大的qps值（最小查询时间对应最佳性能）
    """
    df = load_log_RaBitQ(filepath)
    
    if df.empty:
        return np.array([]), np.array([])
    
    return df['recall'].values, df['qps'].values

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
        n_rows = 1
        fig, axes = plt.subplots(n_rows, 3, figsize=(20, 8), sharex=False, sharey=False)

        fig.subplots_adjust(hspace=0.3, wspace=0.2, top=0.85, left=0.08, right=0.95, bottom=0.12)
        axes = axes.flatten()

        for idx, C in enumerate(Clist):
            ax = axes[idx]
            for i in range(10):
                if i == 0:
                    label = "IVF"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/索引32/recall@10/ivf/nosimd/instructorxl-arxiv-768_100k/instructorxl-arxiv-768_100k_ad_ivf_{C}_0_dist_time.log"
                    recall, Qps = load_log_simple(filename)
                elif i == 1:
                    label = "IVF-ADS"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/索引32/recall@10/ivf/nosimd/instructorxl-arxiv-768_100k/instructorxl-arxiv-768_100k_ad_ivf_{C}_1_dist_time.log"
                    recall, Qps = load_log_simple(filename)
                elif i == 2:
                    label = "IVF-DADE"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/DADEnew/索引/results/nosimd-IVF/instructorxl-arxiv-768_100k_IVF_C{C}_K10_S0.30_P32_3_dist_time.log"
                    recall, Qps = load_log_simple(filename)
                elif i == 3:
                    label = r"IVF-DDC$_{res}$"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/索引32/recall@10/ivf/nosimd/instructorxl-arxiv-768_100k/instructorxl-arxiv-768_100k_ad_ivf_{C}_5_dist_time.log"
                    recall, Qps = load_log_simple(filename)
                elif i == 4:
                    label = r"IVF-DDC$_{pca}$"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/索引32/recall@10/ivf/nosimd/instructorxl-arxiv-768_100k/instructorxl-arxiv-768_100k_ad_ivf_{C}_3_dist_time.log"
                    recall, Qps = load_log_simple(filename)
                elif i == 5:
                    label = r"IVF-DDC$_{opq}$"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/索引32/recall@10/ivf/nosimd/instructorxl-arxiv-768_100k/instructorxl-arxiv-768_100k_ad_ivf_{C}_6_dist_time.log"
                    recall, Qps = load_log_simple(filename)
                elif i == 8:
                    label = "IVF-PDX-ADS"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/PDXnew/索引大小/IVF_PDX_ADSAMPLING{C}_nosimd.csv"
                    recall, Qps = load_result_data_pdx(filename, "instructorxl-arxiv-768_100k")
                elif i == 9:
                    label = "IVF-PDX-BOND"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/PDXnew/索引大小/IVF_PDX_BOND{C}_nosimd.csv"
                    recall, Qps = load_result_data_pdx(filename, "instructorxl-arxiv-768_100k")
                elif i == 6:
                    continue
                elif i == 7:
                    continue

                mask = recall >= 0.80      
                ax.plot(recall[mask], Qps[mask], marker=ivf_marker[i], c=col[i], label=label, alpha=0.5, linestyle="--", markerfacecolor='white', markersize=6, linewidth=2.5, markeredgecolor=col[i], markeredgewidth=1.5)
            

            ax.set_title(f"C={C}", fontsize=22, fontfamily='Arial Unicode MS')
            ax.grid(linestyle='--', linewidth=0.5)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
            ax.set_yscale('log')
            ax.set_ylim(bottom=1)
            ax.set_xlim(left=0.80)
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            ax.tick_params(axis='both', which='major', labelsize=16)

        # 设置公共x/y轴标签
        fig.text(0.5, 0.04, 'Recall@10', ha='center', fontsize=22)
        fig.text(0.04, 0.5, 'Qps', va='center', rotation='vertical', fontsize=22)

        handles, labels = ax.get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=22, 
                          bbox_to_anchor=(0.5, 1.03), handlelength=2.5, handletextpad=1.2, columnspacing=1.5)
        legend.get_frame().set_edgecolor('none')  # 删除图例外边框
        legend.get_frame().set_facecolor('none')  # 删除图例背景


        # plt.rc('font', family='Arial Unicode MS')
        plt.savefig(f'E:/cppwork/dco_benchmarks/DATA/figure/索引大小/IVF_all_nosimd_CLIST.pdf', dpi=400, bbox_inches='tight',format='pdf')
        plt.show()

    for K in [10]:
        n_rows = 1
        fig, axes = plt.subplots(n_rows, 2, figsize=(10, 3.5),sharex=False, sharey=False)

        fig.subplots_adjust(hspace=0.3, wspace=0.2, top=0.85, left=0.08, right=1, bottom=0.12)
        axes = axes.flatten()

        for idx, C in enumerate(Clist):
            ax = axes[idx]
            for i in range(10):
                if i == 0:
                    label = "IVF"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/索引32/recall@10/ivf/simd/instructorxl-arxiv-768_100k/instructorxl-arxiv-768_100k_ad_ivf_{C}_0_dist_time.log"
                    recall, Qps = load_log_simple(filename)
                elif i == 1:
                    label = "IVF-ADS"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/索引32/recall@10/ivf/simd/instructorxl-arxiv-768_100k/instructorxl-arxiv-768_100k_ad_ivf_{C}_1_dist_time.log"
                    recall, Qps = load_log_simple(filename)
                elif i == 2:
                    label = "IVF-DADE"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/DADEnew/索引/results/simd-IVF/instructorxl-arxiv-768_100k_IVF_C{C}_K10_S0.30_P32_3_dist_time.log"
                    recall, Qps = load_log_simple(filename)
                elif i == 3:
                    label = r"IVF-DDC$_{res}$"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/索引32/recall@10/ivf/simd/instructorxl-arxiv-768_100k/instructorxl-arxiv-768_100k_ad_ivf_{C}_5_dist_time.log"
                    recall, Qps = load_log_simple(filename)
                elif i == 4:
                    label = r"IVF-DDC$_{pca}$"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/索引32/recall@10/ivf/simd/instructorxl-arxiv-768_100k/instructorxl-arxiv-768_100k_ad_ivf_{C}_3_dist_time.log"
                    recall, Qps = load_log_simple(filename)
                elif i == 5:
                    label = r"IVF-DDC$_{opq}$"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/索引32/recall@10/ivf/simd/instructorxl-arxiv-768_100k/instructorxl-arxiv-768_100k_ad_ivf_{C}_6_dist_time.log"
                    recall, Qps = load_log_simple(filename)
                elif i == 8:
                    label = "IVF-PDX-ADS"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/PDXnew/索引大小/IVF_PDX_ADSAMPLING{C}_simd.csv"
                    recall, Qps = load_result_data_pdx(filename, "instructorxl-arxiv-768_100k")
                elif i == 9:
                    label = "IVF-PDX-BOND"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/PDXnew/索引大小/IVF_PDX_BOND{C}_simd.csv"
                    recall, Qps = load_result_data_pdx(filename, "instructorxl-arxiv-768_100k")
                elif i == 6:
                    label = "IVF-RaBitQ"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/RabitQnew/索引/instructorxl-arxiv-768_100k/ivf_rabitq_querying_{C}_7.log"
                    recall, Qps = load_log_simple_RaBitQ(filename)
                elif i == 7:
                    continue

                mask = recall >= 0.80      
                ax.plot(recall[mask], Qps[mask], marker=ivf_marker[i], c=col[i], label=label, alpha=0.5, linestyle="--", markerfacecolor='white', markersize=6, linewidth=2.5, markeredgecolor=col[i], markeredgewidth=1.5)
            

            ax.set_title(f"C={C}", fontsize=18, fontfamily='Arial Unicode MS')
            # ax.grid(linestyle='--', linewidth=0.5)
            ax.grid(True, which="major", linestyle="--", linewidth=0.5)  # 主刻度和次刻度都画网格
            ax.minorticks_on()
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
            ax.set_yscale('log')
            ax.set_ylim(bottom=185,top=8800)
            ax.set_yticks([1e3,1e4]) 
            ax.set_xlim(left=0.80)
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            ax.tick_params(axis='both', which='major', labelsize=16)

        # 设置公共x/y轴标签
        fig.text(0.53, -0.03, 'Recall@10', ha='center', fontsize=18)
        fig.text(0.01, 0.48, 'QPS',
            va='center', rotation='vertical', fontsize=18
        )

        handles, labels = ax.get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=18, 
                          bbox_to_anchor=(0.5, 1.29), handlelength=2.5, handletextpad=1.2, columnspacing=1)
        legend.get_frame().set_edgecolor('none')  # 删除图例外边框
        legend.get_frame().set_facecolor('none')  # 删除图例背景


        # plt.rc('font', family='Arial Unicode MS')
        plt.savefig(f'E:/cppwork/dco_benchmarks/DATA/figure/索引大小/IVF_all_simd_CLIST.pdf', dpi=400, bbox_inches='tight',format='pdf')
        plt.show()

    for K in [10]:
        n_rows = 1
        fig, axes = plt.subplots(n_rows, 3, figsize=(20, 8), sharex=False, sharey=False)

        fig.subplots_adjust(hspace=0.3, wspace=0.2, top=0.85, left=0.08, right=0.95, bottom=0.12)
        axes = axes.flatten()

        for idx, M in enumerate(Mlist):
            ax = axes[idx]
            for i in range(10):
                if i == 0:
                    label = "HNSW"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/索引32/recall@10/hnsw/nosimd/instructorxl-arxiv-768_100k/instructorxl-arxiv-768_100k_ad_hnsw_ef500_M{M}_0_dist_time.log"
                    recall, Qps = load_log_simple(filename)
                elif i == 1:
                    label = "HNSW-ADS"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/索引32/recall@10/hnsw/nosimd/instructorxl-arxiv-768_100k/instructorxl-arxiv-768_100k_ad_hnsw_ef500_M{M}_1_dist_time.log"
                    recall, Qps = load_log_simple(filename)
                elif i == 2:
                    label = "HNSW-DADE"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/DADEnew/索引/results/nosimd-HNSW/instructorxl-arxiv-768_100k_HNSW_ef500_M{M}_K10_S0.30_P32_3_dist_time.log"
                    recall, Qps = load_log_simple(filename)
                elif i == 3:
                    label = r"HNSW-DDC$_{res}$"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/索引32/recall@10/hnsw/nosimd/instructorxl-arxiv-768_100k/instructorxl-arxiv-768_100k_ad_hnsw_ef500_M{M}_7_dist_time.log"
                    recall, Qps = load_log_simple(filename)
                elif i == 4:
                    label = r"HNSW-DDC$_{pca}$"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/索引32/recall@10/hnsw/nosimd/instructorxl-arxiv-768_100k/instructorxl-arxiv-768_100k_ad_hnsw_ef500_M{M}_6_dist_time.log"
                    recall, Qps = load_log_simple(filename)
                elif i == 5:
                    label = r"HNSW-DDC$_{opq}$"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/索引32/recall@10/hnsw/nosimd/instructorxl-arxiv-768_100k/instructorxl-arxiv-768_100k_ad_hnsw_ef500_M{M}_4_dist_time.log"
                    recall, Qps = load_log_simple(filename)
                elif i == 8:
                    continue
                elif i == 9:
                    continue
                elif i == 6:
                    continue
                elif i == 7:
                    continue

                mask = recall >= 0.80      
                ax.plot(recall[mask], Qps[mask], marker=ivf_marker[i], c=col[i], label=label, alpha=0.5, linestyle="--", markerfacecolor='white', markersize=6, linewidth=2.5, markeredgecolor=col[i], markeredgewidth=1.5)
            

            ax.set_title(f"M={M}", fontsize=22, fontfamily='Arial Unicode MS')
            ax.grid(linestyle='--', linewidth=0.5)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
            ax.set_yscale('log')
            ax.set_ylim(bottom=1)
            ax.set_xlim(left=0.80)
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            ax.tick_params(axis='both', which='major', labelsize=16)

        # 设置公共x/y轴标签
        fig.text(0.5, 0.04, 'Recall@10', ha='center', fontsize=22)
        fig.text(0.04, 0.5, 'Qps', va='center', rotation='vertical', fontsize=22)

        handles, labels = ax.get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=22, 
                          bbox_to_anchor=(0.5, 1.03), handlelength=2.5, handletextpad=1.2, columnspacing=1.5)
        legend.get_frame().set_edgecolor('none')  # 删除图例外边框
        legend.get_frame().set_facecolor('none')  # 删除图例背景


        # plt.rc('font', family='Arial Unicode MS')
        plt.savefig(f'E:/cppwork/dco_benchmarks/DATA/figure/索引大小/HNSW_all_nosimd_M.pdf', dpi=400, bbox_inches='tight',format='pdf')
        plt.show()

    for K in [10]:
        n_rows = 1
        fig, axes = plt.subplots(n_rows, 3, figsize=(20, 8), sharex=False, sharey=False)

        fig.subplots_adjust(hspace=0.3, wspace=0.2, top=0.85, left=0.08, right=0.95, bottom=0.12)
        axes = axes.flatten()

        for idx, EF in enumerate(EFlist):
            ax = axes[idx]
            for i in range(10):
                if i == 0:
                    label = "HNSW"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/索引32/recall@10/hnsw/nosimd/instructorxl-arxiv-768_100k/instructorxl-arxiv-768_100k_ad_hnsw_ef{EF}_M16_0_dist_time.log"
                    recall, Qps = load_log_simple(filename)
                elif i == 1:
                    label = "HNSW-ADS"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/索引32/recall@10/hnsw/nosimd/instructorxl-arxiv-768_100k/instructorxl-arxiv-768_100k_ad_hnsw_ef{EF}_M16_1_dist_time.log"
                    recall, Qps = load_log_simple(filename)
                elif i == 2:
                    label = "HNSW-DADE"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/DADEnew/索引/results/nosimd-HNSW/instructorxl-arxiv-768_100k_HNSW_ef{EF}_M16_K10_S0.30_P32_3_dist_time.log"
                    recall, Qps = load_log_simple(filename)
                elif i == 3:
                    label = r"HNSW-DDC$_{res}$"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/索引32/recall@10/hnsw/nosimd/instructorxl-arxiv-768_100k/instructorxl-arxiv-768_100k_ad_hnsw_ef{EF}_M16_7_dist_time.log"
                    recall, Qps = load_log_simple(filename)
                elif i == 4:
                    label = r"HNSW-DDC$_{pca}$"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/索引32/recall@10/hnsw/nosimd/instructorxl-arxiv-768_100k/instructorxl-arxiv-768_100k_ad_hnsw_ef{EF}_M16_6_dist_time.log"
                    recall, Qps = load_log_simple(filename)
                elif i == 5:
                    label = r"HNSW-DDC$_{opq}$"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/索引32/recall@10/hnsw/nosimd/instructorxl-arxiv-768_100k/instructorxl-arxiv-768_100k_ad_hnsw_ef{EF}_M16_4_dist_time.log"
                    recall, Qps = load_log_simple(filename)
                elif i == 8:
                    continue
                elif i == 9:
                    continue
                elif i == 6:
                    continue
                elif i == 7:
                    continue

                mask = recall >= 0.80      
                ax.plot(recall[mask], Qps[mask], marker=ivf_marker[i], c=col[i], label=label, alpha=0.5, linestyle="--", markerfacecolor='white', markersize=6, linewidth=2.5, markeredgecolor=col[i], markeredgewidth=1.5)
            

            ax.set_title(f"EF={EF}", fontsize=22, fontfamily='Arial Unicode MS')
            ax.grid(linestyle='--', linewidth=0.5)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
            ax.set_yscale('log')
            ax.set_ylim(bottom=1)
            ax.set_xlim(left=0.80)
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            ax.tick_params(axis='both', which='major', labelsize=16)

        # 设置公共x/y轴标签
        fig.text(0.5, 0.04, 'Recall@10', ha='center', fontsize=22)
        fig.text(0.04, 0.5, 'Qps', va='center', rotation='vertical', fontsize=22)

        handles, labels = ax.get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=22, 
                          bbox_to_anchor=(0.5, 1.03), handlelength=2.5, handletextpad=1.2, columnspacing=1.5)
        legend.get_frame().set_edgecolor('none')  # 删除图例外边框
        legend.get_frame().set_facecolor('none')  # 删除图例背景


        # plt.rc('font', family='Arial Unicode MS')
        plt.savefig(f'E:/cppwork/dco_benchmarks/DATA/figure/索引大小/HNSW_all_nosimd_EF.pdf', dpi=400, bbox_inches='tight',format='pdf')
        plt.show()


    for K in [10]:
        n_rows = 1
        fig, axes = plt.subplots(n_rows, 3, figsize=(20, 8), sharex=False, sharey=False)

        fig.subplots_adjust(hspace=0.3, wspace=0.2, top=0.85, left=0.08, right=0.95, bottom=0.12)
        axes = axes.flatten()

        for idx, M in enumerate(Mlist):
            ax = axes[idx]
            for i in range(10):
                if i == 0:
                    label = "HNSW"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/索引32/recall@10/hnsw/simd/instructorxl-arxiv-768_100k/instructorxl-arxiv-768_100k_ad_hnsw_ef500_M{M}_0_dist_time.log"
                    recall, Qps = load_log_simple(filename)
                elif i == 1:
                    label = "HNSW-ADS"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/索引32/recall@10/hnsw/simd/instructorxl-arxiv-768_100k/instructorxl-arxiv-768_100k_ad_hnsw_ef500_M{M}_1_dist_time.log"
                    recall, Qps = load_log_simple(filename)
                elif i == 2:
                    label = "HNSW-DADE"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/DADEnew/索引/results/simd-HNSW/instructorxl-arxiv-768_100k_HNSW_ef500_M{M}_K10_S0.30_P32_3_dist_time.log"
                    recall, Qps = load_log_simple(filename)
                elif i == 3:
                    label = r"HNSW-DDC$_{res}$"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/索引32/recall@10/hnsw/simd/instructorxl-arxiv-768_100k/instructorxl-arxiv-768_100k_ad_hnsw_ef500_M{M}_7_dist_time.log"
                    recall, Qps = load_log_simple(filename)
                elif i == 4:
                    label = r"HNSW-DDC$_{pca}$"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/索引32/recall@10/hnsw/simd/instructorxl-arxiv-768_100k/instructorxl-arxiv-768_100k_ad_hnsw_ef500_M{M}_6_dist_time.log"
                    recall, Qps = load_log_simple(filename)
                elif i == 5:
                    label = r"HNSW-DDC$_{opq}$"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/索引32/recall@10/hnsw/simd/instructorxl-arxiv-768_100k/instructorxl-arxiv-768_100k_ad_hnsw_ef500_M{M}_4_dist_time.log"
                    recall, Qps = load_log_simple(filename)
                elif i == 8:
                    continue
                elif i == 9:
                    continue
                elif i == 6:
                    label = r"HNSW-RaBitQ"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/RabitQnew/索引/instructorxl-arxiv-768_100k/hnsw_rabitq_querying_ef500_M{M}_7.log"
                    recall, Qps = load_log_simple_RaBitQ(filename)
                elif i == 7:
                    continue

                mask = recall >= 0.80      
                ax.plot(recall[mask], Qps[mask], marker=ivf_marker[i], c=col[i], label=label, alpha=0.5, linestyle="--", markerfacecolor='white', markersize=6, linewidth=2.5, markeredgecolor=col[i], markeredgewidth=1.5)
            

            ax.set_title(f"M={M}", fontsize=22, fontfamily='Arial Unicode MS')
            ax.grid(linestyle='--', linewidth=0.5)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
            ax.set_yscale('log')
            ax.set_ylim(bottom=1)
            ax.set_xlim(left=0.80)
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            ax.tick_params(axis='both', which='major', labelsize=16)

        # 设置公共x/y轴标签
        fig.text(0.5, 0.04, 'Recall@10', ha='center', fontsize=22)
        fig.text(0.04, 0.5, 'Qps', va='center', rotation='vertical', fontsize=22)

        handles, labels = ax.get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=22, 
                          bbox_to_anchor=(0.5, 1.03), handlelength=2.5, handletextpad=1.2, columnspacing=1.5)
        legend.get_frame().set_edgecolor('none')  # 删除图例外边框
        legend.get_frame().set_facecolor('none')  # 删除图例背景


        # plt.rc('font', family='Arial Unicode MS')
        plt.savefig(f'E:/cppwork/dco_benchmarks/DATA/figure/索引大小/HNSW_all_simd_M.pdf', dpi=400, bbox_inches='tight',format='pdf')
        plt.show()

    for K in [10]:
        n_rows = 1
        fig, axes = plt.subplots(n_rows, 3, figsize=(20, 8), sharex=False, sharey=False)

        fig.subplots_adjust(hspace=0.3, wspace=0.2, top=0.85, left=0.08, right=0.95, bottom=0.12)
        axes = axes.flatten()

        for idx, EF in enumerate(EFlist):
            ax = axes[idx]
            for i in range(10):
                if i == 0:
                    label = "HNSW"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/索引32/recall@10/hnsw/simd/instructorxl-arxiv-768_100k/instructorxl-arxiv-768_100k_ad_hnsw_ef{EF}_M16_0_dist_time.log"
                    recall, Qps = load_log_simple(filename)
                elif i == 1:
                    label = "HNSW-ADS"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/索引32/recall@10/hnsw/simd/instructorxl-arxiv-768_100k/instructorxl-arxiv-768_100k_ad_hnsw_ef{EF}_M16_1_dist_time.log"
                    recall, Qps = load_log_simple(filename)
                elif i == 2:
                    label = "HNSW-DADE"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/DADEnew/索引/results/simd-HNSW/instructorxl-arxiv-768_100k_HNSW_ef{EF}_M16_K10_S0.30_P32_3_dist_time.log"
                    recall, Qps = load_log_simple(filename)
                elif i == 3:
                    label = r"HNSW-DDC$_{res}$"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/索引32/recall@10/hnsw/simd/instructorxl-arxiv-768_100k/instructorxl-arxiv-768_100k_ad_hnsw_ef{EF}_M16_7_dist_time.log"
                    recall, Qps = load_log_simple(filename)
                elif i == 4:
                    label = r"HNSW-DDC$_{pca}$"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/索引32/recall@10/hnsw/simd/instructorxl-arxiv-768_100k/instructorxl-arxiv-768_100k_ad_hnsw_ef{EF}_M16_6_dist_time.log"
                    recall, Qps = load_log_simple(filename)
                elif i == 5:
                    label = r"HNSW-DDC$_{opq}$"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Resnew/索引32/recall@10/hnsw/simd/instructorxl-arxiv-768_100k/instructorxl-arxiv-768_100k_ad_hnsw_ef{EF}_M16_4_dist_time.log"
                    recall, Qps = load_log_simple(filename)
                elif i == 8:
                    continue
                elif i == 9:
                    continue
                elif i == 6:
                    label = r"HNSW-RaBitQ"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/RabitQnew/索引/instructorxl-arxiv-768_100k/hnsw_rabitq_querying_ef{EF}_M16_7.log"
                    recall, Qps = load_log_simple_RaBitQ(filename)
                elif i == 7:
                    continue

                mask = recall >= 0.80      
                ax.plot(recall[mask], Qps[mask], marker=ivf_marker[i], c=col[i], label=label, alpha=0.5, linestyle="--", markerfacecolor='white', markersize=6, linewidth=2.5, markeredgecolor=col[i], markeredgewidth=1.5)
            

            ax.set_title(f"EF={EF}", fontsize=22, fontfamily='Arial Unicode MS')
            ax.grid(linestyle='--', linewidth=0.5)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
            ax.set_yscale('log')
            ax.set_ylim(bottom=1)
            ax.set_xlim(left=0.80)
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            ax.tick_params(axis='both', which='major', labelsize=16)

        # 设置公共x/y轴标签
        fig.text(0.5, 0.04, 'Recall@10', ha='center', fontsize=22)
        fig.text(0.04, 0.5, 'Qps', va='center', rotation='vertical', fontsize=22)

        handles, labels = ax.get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=22, 
                          bbox_to_anchor=(0.5, 1.03), handlelength=2.5, handletextpad=1.2, columnspacing=1.5)
        legend.get_frame().set_edgecolor('none')  # 删除图例外边框
        legend.get_frame().set_facecolor('none')  # 删除图例背景


        # plt.rc('font', family='Arial Unicode MS')
        plt.savefig(f'E:/cppwork/dco_benchmarks/DATA/figure/索引大小/HNSW_all_simd_EF.pdf', dpi=400, bbox_inches='tight',format='pdf')
        plt.show()
