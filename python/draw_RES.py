import numpy as np
import matplotlib.pyplot as plt
from utils import fvecs_read, fvecs_write
import os
import struct
from tqdm import tqdm
import pandas as pd
import matplotlib.ticker as ticker

datasets = ['nytimes-16-angular', 'glove-50-angular','glove-200-angular','sift-128-euclidean','msong-420','contriever-768','gist-960-euclidean','deep-image-96-angular','instructorxl-arxiv-768','openai-1536-angular']

hnsw_marker = ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o' ]
ivf_marker = ['s', 's', 's', 's', 's', 's', 's', 's', 's', 's', 's', 's']
col = ['r', 'b', 'y', 'g', 'c', 'm', 'olive', 'gold', 'violet', 'pink', 'brown', 'gray']


def load_result_data(filename, dataset):
    df = pd.read_csv(filename)
    # Filter data for specific dataset
    df_dataset = df[df['dataset'] == dataset]
    # Get recall and QPS data
    recall = df_dataset['recall'].values/100
    # Calculate QPS (queries per second)
    Qps = 1e9 / df_dataset['per_query_us'].values  # Convert average query time (ms) to QPS
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
    for dataset in datasets:
        print(f"visual - {dataset}")
        for K in [10]:
            plt.figure(figsize=(12, 8))
            for i in range(6):
                filename = f"E:/cppwork/dco_benchmarks/DATA/Res/avx512/HNSW_RES_avx512_0.csv"
                label = "null"
                if i == 0:
                    label = "hnsw-simd"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Res/avx512/HNSW_RES_avx512_0.csv"
                elif i == 1:
                    label = "hnsw-adsampling-simd"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Res/avx512/HNSW_RES_avx512_1.csv"
                elif i == 2:
                    label = "hnsw-bsa6-simd"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Res/avx512/HNSW_RES_avx512_6.csv"
                elif i == 3:
                    label = "hnsw-bsa7-simd"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Res/avx512/HNSW_RES_avx512_7.csv" 
                elif i == 4:
                    label = "hnsw-bsa8-simd"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Res/avx512/HNSW_RES_avx512_8.csv" 
                elif i == 5:
                    label = "hnsw-finger-simd"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/Res/avx512/HNSW_RES_avx512_9.csv" 

                recall, Qps = load_result_data(filename, dataset)
                # 只保留recall >= 0.80的数据点
                mask = recall >= 0.80
                plt.plot(recall[mask], Qps[mask], marker=ivf_marker[i], c=col[i], label=label, alpha=0.5, linestyle="--", markerfacecolor='white')

            plt.xlabel("Recall@10")
            plt.ylabel("Qps")
            plt.legend(loc="upper right")
            plt.grid(linestyle='--', linewidth=0.5)
            ax = plt.gca()
            

            # 设置y轴使用科学计数法
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            # 设置y轴刻度标签的格式
            ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
            # 去掉y轴0刻度
            ax.set_ylim(bottom=0)
            # 设置y轴刻度数量
            ax.yaxis.set_major_locator(plt.MaxNLocator(5))
            # 设置x轴范围从0.80开始
            ax.set_xlim(left=0.80)
            
            plt.rc('font', family='Times New Roman')
            plt.savefig(f'E:/cppwork/dco_benchmarks/DATA/figure/RES/HNSW/{dataset}_hnsw_qps_@{K}.png', dpi=400)
            plt.show()
