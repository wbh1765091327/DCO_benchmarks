import os
import pandas as pd
import glob
import re

def extract_dataset_name_from_path(filepath):
    """从文件路径中提取数据集名称（从目录名）"""
    # 获取文件的父目录名作为数据集名称
    directory = os.path.dirname(filepath)
    dataset = os.path.basename(directory)
    return dataset

def extract_method_from_filename(filename):
    """从文件名中提取方法类型"""
    filename_lower = filename.lower()
    if 'hnsw' in filename_lower:
        return 'hnsw'
    elif 'ivf' in filename_lower:
        return 'ivf'
    elif 'symqg' in filename_lower:
        return 'symqg'
    else:
        return 'unknown'

def extract_bit_from_filename(filename):
    """从文件名中提取bit标识"""
    # 使用正则表达式匹配文件名末尾的数字（bit标识）
    # 例如：hnsw_rabitq_querying_7.log -> 7
    #       hnsw_rabitq_querying_7_nosimd.log -> 7
    match = re.search(r'_(\d+)(?:_nosimd)?\.log$', filename)
    if match:
        return int(match.group(1))
    else:
        return None

def process_log_file(filepath):
    """处理单个log文件并返回DataFrame"""
    # 从路径中提取数据集名称
    dataset = extract_dataset_name_from_path(filepath)
    # 从文件名中提取方法类型和bit标识
    filename = os.path.basename(filepath)
    method = extract_method_from_filename(filename)
    bit = extract_bit_from_filename(filename)
    
    # 读取log文件
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                parts = line.split()
                if len(parts) == 3:  # 只有3列：efsearch, recall, per_query_us
                    efsearch = int(parts[0])
                    recall = float(parts[1])
                    per_query_us = float(parts[2])
                    
                    data.append({
                        'dataset': dataset,
                        'method': method,
                        'bit': bit,
                        'efsearch': efsearch,
                        'recall': recall,
                        'per_query_us': per_query_us
                    })
    
    return pd.DataFrame(data)

def consolidate_logs_by_method_and_bit(directory_path, output_dir):
    """按方法和bit分别整合log文件到不同的CSV文件"""
    # 递归获取所有.log文件
    log_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.log'):
                log_files.append(os.path.join(root, file))
    
    if not log_files:
        print(f"在目录 {directory_path} 及其子目录中没有找到.log文件")
        return
    
    print(f"找到 {len(log_files)} 个log文件")
    
    # 按方法和bit分组文件
    method_bit_files = {}
    
    for log_file in log_files:
        filename = os.path.basename(log_file)
        
        # 只处理 nosimd 文件，跳过 SIMD 文件
        if '_nosimd' not in filename.lower():
            continue
        
        method = extract_method_from_filename(filename)
        bit = extract_bit_from_filename(filename)
        
        if method != 'unknown' and bit is not None:
            key = f"{method}_{bit}bit_nosimd"
            if key not in method_bit_files:
                method_bit_files[key] = []
            method_bit_files[key].append(log_file)
        else:
            print(f"警告: 无法识别方法类型或bit标识，文件: {filename}")
    
    # 为每种方法+bit组合处理文件
    for method_bit, files in method_bit_files.items():
        if not files:
            print(f"没有找到 {method_bit} 的文件")
            continue
            
        print(f"\n处理 {method_bit}，找到 {len(files)} 个文件")
        
        # 按数据集分组处理文件
        dataset_files = {}
        for log_file in files:
            dataset = extract_dataset_name_from_path(log_file)
            if dataset not in dataset_files:
                dataset_files[dataset] = []
            dataset_files[dataset].append(log_file)
        
        print(f"发现的数据集 ({method_bit}):")
        for dataset, file_list in dataset_files.items():
            print(f"  {dataset}: {len(file_list)} 个文件")
        
        # 处理所有文件
        all_data = []
        for dataset, file_list in dataset_files.items():
            print(f"  处理数据集: {dataset}")
            for log_file in file_list:
                print(f"    处理文件: {os.path.basename(log_file)}")
                df = process_log_file(log_file)
                all_data.append(df)
        
        # 合并所有数据
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # 按dataset和efsearch分组，保留per_query_us最小的行（去重）
            print(f"  原始数据行数: {len(combined_df)}")
            combined_df = combined_df.loc[combined_df.groupby(['dataset', 'efsearch'])['per_query_us'].idxmin()]
            print(f"  去重后数据行数: {len(combined_df)}")
            
            # 保存到CSV文件
            output_file = os.path.join(output_dir, f"{method_bit.upper()}_RabitQ_nosimd.csv")
            combined_df.to_csv(output_file, index=False)
            print(f"  {method_bit} 数据已保存到: {output_file}")
            print(f"  总共 {len(combined_df)} 行数据")
            
            # 显示数据集统计信息
            print(f"  {method_bit} 数据集统计:")
            print(combined_df['dataset'].value_counts())
            
            print(f"\n{method_bit} 每个数据集的efsearch值范围:")
            for dataset in combined_df['dataset'].unique():
                dataset_data = combined_df[combined_df['dataset'] == dataset]
                efsearch_values = sorted(dataset_data['efsearch'].unique())
                print(f"  {dataset}: efsearch值 {efsearch_values}")

if __name__ == "__main__":
    # 设置路径
    log_directory = r"E:\cppwork\dco_benchmarks\DATA\RabitQnew"
    output_directory = r"E:\cppwork\dco_benchmarks\DATA\RabitQnew"
    
    # 执行整合（按方法和bit分别处理）
    print("开始按方法和bit分别整合log文件...")
    consolidate_logs_by_method_and_bit(log_directory, output_directory)
    
    print("\n所有方法和bit的数据整合完成！") 