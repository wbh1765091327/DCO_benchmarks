import numpy as np
import matplotlib.pyplot as plt
from utils import fvecs_read, fvecs_write
import os
import struct
from tqdm import tqdm
import pandas as pd


datasets = ['nytimes-16-angular', 'glove-50-angular','glove-200-angular','sift-128-euclidean','msong-420','contriever-768','gist-960-euclidean','deep-image-96-angular','instructorxl-arxiv-768','openai-1536-angular']

hnsw_marker = ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o' ]
ivf_marker = ['s', 's', 's', 's', 's', 's', 's', 's', 's', 's', 's', 's']
col = ['r', 'b', 'y', 'g', 'c', 'm', 'olive', 'gold', 'violet', 'pink', 'brown', 'gray']


def load_result_data(filename, dataset):
    df = pd.read_csv(filename)
    # Filter data for specific dataset
    df_dataset = df[df['dataset'] == dataset]
    # Get recall and QPS data
    recall = df_dataset['recall'].values
    # Calculate QPS (queries per second)
    Qps = 1e6 / df_dataset['avg_all'].values  # Convert average query time (ms) to QPS
    return recall, Qps


if __name__ == "__main__":

    for dataset in datasets:
        print(f"visual - {dataset}")
        for K in [10]:
            plt.figure(figsize=(12, 8))
            for i in range(12):
                filename = f"E:/cppwork/dco_benchmarks/DATA/PDX/IVF_FAISS.csv"
                label = "null"
                if i == 0:
                    label = "ivf-nary-adsampling"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/PDX/IVF_NARY_ADSAMPLING.csv"
                elif i == 1:
                    label = "ivf-nary-adsampling-simd"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/PDX/IVF_NARY_ADSAMPLING_SIMD.csv"
                elif i == 2:
                    label = "ivf-nary-bsa"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/PDX/IVF_NARY_BSA.csv"
                elif i == 3:
                    label = "ivf-nary-bsa-simd"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/PDX/IVF_NARY_BSA_SIMD.csv"
                elif i == 4:
                    label = "ivf-pdx-adsampling"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/PDX/IVF_PDX_ADSAMPLING.csv"
                elif i == 5:
                    label = "ivf-pdx-bsa"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/PDX/IVF_PDX_BSA.csv"
                elif i == 6:
                    label = "ivf-pdx-bond-dec"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/PDX/pdx-bond-dec.csv"
                elif i == 7:
                    label = "ivf-pdx-bond-decimpr"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/PDX/pdx-bond-decimpr.csv"
                elif i == 8:
                    label = "ivf-pdx-bond-dtm"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/PDX/pdx-bond-dtm.csv"
                elif i == 9:
                    label = "ivf-pdx-bond-dz"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/PDX/pdx-bond-dz.csv"
                elif i == 10:
                    label = "ivf-pdx-bond-sec"
                    filename = f"E:/cppwork/dco_benchmarks/DATA/PDX/pdx-bond-sec.csv"
                elif i == 11:
                    label = "ivf-pdx-bond"       
                    filename = f"E:/cppwork/dco_benchmarks/DATA/PDX/pdx-bond.csv"             

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
            plt.savefig(f'E:/cppwork/dco_benchmarks/DATA/figure/{dataset}_ivf_qps_@{K}.png', dpi=400)
            plt.show()