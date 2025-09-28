import os
import pandas as pd
import glob
import re

def extract_dataset_name_hnsw(filename):
    """从文件名中提取数据集名称"""
    # 移除文件扩展名
    name = os.path.splitext(filename)[0]
    # 移除HNSW相关的后缀
    name = re.sub(r'_HNSW_ef\d+_M\d+_K\d+_S[\d.]+_P\d+_\d_dist_time$', '', name)
    return name

def extract_dataset_name_ivf(filename):
    """从文件名中提取数据集名称"""
    # 移除文件扩展名
    name = os.path.splitext(filename)[0]
    # 移除IVF相关的后缀
    name = re.sub(r'_IVF_C\d+_K\d+_S[\d.]+_P\d+_\d_dist_time$', '', name)
    return name

# def process_log_file(filepath):
#     """处理单个log文件并返回DataFrame"""
#     # 提取数据集名称
#     filename = os.path.basename(filepath)
#     dataset = extract_dataset_name_ivf(filename)
    
#     # 读取log文件
#     data = []
#     with open(filepath, 'r', encoding='utf-8') as f:
#         for line in f:
#             line = line.strip()
#             if line:  # 跳过空行
#                 parts = line.split()
#                 if len(parts) == 4:
#                     efsearch = int(parts[0])
#                     recall = float(parts[1])
#                     per_query_us = float(parts[2])
#                     dimensionality = int(parts[3])
                    
#                     data.append({
#                         'dataset': dataset,
#                         'efsearch': efsearch,
#                         'recall': recall,
#                         'per_query_us': per_query_us,
#                         'dimensionality': dimensionality
#                     })
    
#     return pd.DataFrame(data)

def process_log_file_dist_time(filepath):
    """处理单个log文件并返回DataFrame"""
    # 提取数据集名称
    filename = os.path.basename(filepath)
    dataset = extract_dataset_name_ivf(filename)
    
    # 读取log文件
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                parts = line.split()
                if len(parts) == 5:
                    efsearch = int(parts[0])
                    recall = float(parts[1])
                    per_query_us = float(parts[2])
                    total_time = float(parts[3])
                    distance_time = float(parts[4])
                    
                    data.append({
                        'dataset': dataset,
                        'efsearch': efsearch,
                        'recall': recall,
                        'per_query_us': per_query_us,
                        'total_time': total_time,
                        'distance_time': distance_time
                    })
    
    return pd.DataFrame(data)

def consolidate_logs(directory_path, output_csv):
    """整合目录下所有log文件到一个CSV文件"""
    # 获取所有.log文件
    log_files = glob.glob(os.path.join(directory_path, "*dist_time.log"))
    
    if not log_files:
        print(f"在目录 {directory_path} 中没有找到.log文件")
        return
    
    print(f"找到 {len(log_files)} 个log文件")
    
    # 处理所有文件
    all_data = []
    for log_file in log_files:
        print(f"处理文件: {os.path.basename(log_file)}")
        df = process_log_file_dist_time(log_file)
        all_data.append(df)
    
    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 保存到CSV文件
    combined_df.to_csv(output_csv, index=False)
    print(f"数据已保存到: {output_csv}")
    print(f"总共 {len(combined_df)} 行数据")
    
    # 显示数据集统计信息
    print("\n数据集统计:")
    print(combined_df['dataset'].value_counts())
    
    return combined_df

if __name__ == "__main__":
    # 设置路径
    log_directory = r"E:\cppwork\dco_benchmarks\DATA\DADEnew\数据规模分布-32\results\simd-HNSW"
    output_file = r"E:\cppwork\dco_benchmarks\DATA\DADEnew\数据规模分布-32\HNSW_DADE_3_SIMD_dist_time.csv"
    
    # 执行整合
    result_df = consolidate_logs(log_directory, output_file)
    
    if result_df is not None:
        print("\n前5行数据预览:")
        print(result_df.head()) 