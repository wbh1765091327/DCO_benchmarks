import os
import pandas as pd
import glob
import re

def extract_dataset_name_hnsw(filename, type):
    """从文件名中提取数据集名称"""
    # 移除文件扩展名
    name = os.path.splitext(filename)[0]
    # 移除HNSW相关的后缀，包括_ad_hnsw_n.log格式
    # 使用f-string来正确插入type变量，注意转义
    pattern = f'_ad_ivf_{type}_dist_time$'
    name = re.sub(pattern, '', name)
    return name


def process_log_file(filepath, type):
    """处理单个log文件并返回DataFrame"""
    # 提取数据集名称
    filename = os.path.basename(filepath)
    dataset = extract_dataset_name_hnsw(filename, type)
    
    # 读取log文件
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                parts = line.split()
                if len(parts) == 4:
                    efsearch = int(parts[0])
                    recall = float(parts[1])
                    per_query_us = float(parts[2])
                    dimensionality = int(parts[3])
                    
                    data.append({
                        'dataset': dataset,
                        'efsearch': efsearch,
                        'recall': recall,
                        'per_query_us': per_query_us,
                        'dimensionality': dimensionality
                    })
    
    return pd.DataFrame(data)

def consolidate_logs_recursive(directory_path, output_csv, type):
    """递归整合目录下所有子目录中的log文件到一个CSV文件"""
    # 递归获取所有.log文件，但只处理特定type的文件
    log_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            # 只处理匹配特定type的log文件
            if file.endswith(f'_ad_hnsw_{type}.log'):
                log_files.append(os.path.join(root, file))
    
    if not log_files:
        print(f"在目录 {directory_path} 及其子目录中没有找到_ad_hnsw_{type}.log文件")
        return
    
    print(f"找到 {len(log_files)} 个_ad_hnsw_{type}.log文件")
    
    # 按数据集分组处理文件
    dataset_files = {}
    for log_file in log_files:
        filename = os.path.basename(log_file)
        dataset = extract_dataset_name_hnsw(filename, type)
        if dataset not in dataset_files:
            dataset_files[dataset] = []
        dataset_files[dataset].append(log_file)
    
    print(f"发现 {len(dataset_files)} 个数据集:")
    for dataset, files in dataset_files.items():
        print(f"  {dataset}: {len(files)} 个文件")
    
    # 处理所有文件
    all_data = []
    for dataset, files in dataset_files.items():
        print(f"\n处理数据集: {dataset}")
        for log_file in files:
            print(f"  处理文件: {os.path.basename(log_file)}")
            df = process_log_file(log_file, type)
            all_data.append(df)
    
    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 按dataset和efsearch分组，保留per_query_us最小的行（去重）
    print(f"\n原始数据行数: {len(combined_df)}")
    combined_df = combined_df.loc[combined_df.groupby(['dataset', 'efsearch'])['per_query_us'].idxmin()]
    print(f"去重后数据行数: {len(combined_df)}")
    
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
    log_directory = r"E:\cppwork\dco_benchmarks\DATA\Res\nosimd"
    output_file = r"E:\cppwork\dco_benchmarks\DATA\Res\nosimd\HNSW_RES_nosimd_1.csv"
    
    # 执行整合（递归处理所有子目录）
    print("开始递归整合所有子目录中的0.log文件...")
    result_df = consolidate_logs_recursive(log_directory, output_file, 1)
    
    if result_df is not None:
        print("\n前5行数据预览:")
        print(result_df.head())
        
        print("\n每个数据集的efsearch值范围:")
        for dataset in result_df['dataset'].unique():
            dataset_data = result_df[result_df['dataset'] == dataset]
            efsearch_values = sorted(dataset_data['efsearch'].unique())
            print(f"{dataset}: efsearch值 {efsearch_values}") 