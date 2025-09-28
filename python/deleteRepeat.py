import pandas as pd
import os

def remove_duplicate_efsearch(input_file, output_file):
    """
    删除CSV文件中相同efsearch值的重复数据，只保留per_query_us最小的行
    
    Args:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
    """
    # 读取CSV文件
    print(f"正在读取文件: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"原始数据行数: {len(df)}")
    print(f"原始数据列: {list(df.columns)}")
    
    # 显示每个数据集的efsearch值统计
    print("\n原始数据中每个数据集的efsearch值统计:")
    for dataset in df['dataset'].unique():
        dataset_data = df[df['dataset'] == dataset]
        print(f"{dataset}: {len(dataset_data)} 行, efsearch值: {sorted(dataset_data['efsearch'].unique())}")
    
    # 按dataset和efsearch分组，保留per_query_us最小的行
    df_cleaned = df.loc[df.groupby(['dataset', 'efsearch'])['per_query_us'].idxmin()]
    
    print(f"\n清理后数据行数: {len(df_cleaned)}")
    
    # 显示清理后每个数据集的efsearch值统计
    print("\n清理后数据中每个数据集的efsearch值统计:")
    for dataset in df_cleaned['dataset'].unique():
        dataset_data = df_cleaned[df_cleaned['dataset'] == dataset]
        print(f"{dataset}: {len(dataset_data)} 行, efsearch值: {sorted(dataset_data['efsearch'].unique())}")
    
    # 保存清理后的数据
    df_cleaned.to_csv(output_file, index=False)
    print(f"\n清理后的数据已保存到: {output_file}")
    
    # 显示删除的行数
    removed_rows = len(df) - len(df_cleaned)
    print(f"删除了 {removed_rows} 行重复数据")
    
    return df_cleaned

def process_multiple_files():
    """处理多个CSV文件"""
    base_dir = r"E:\cppwork\dco_benchmarks\DATA\DADE"
    files_to_process = [
        "HNSW_DADE_3_SIMD.csv",
        "HNSW_DADE_3.csv", 
        "HNSW_DADE_4.csv",
        "HNSW_DADE_4_SIMD.csv",
        "IVF_DADE_3.csv",
        "IVF_DADE_3_SIMD.csv", 
        "IVF_DADE_4.csv",
        "IVF_DADE_4_SIMD.csv"
    ]
    
    for filename in files_to_process:
        input_path = os.path.join(base_dir, filename)
        output_path = os.path.join(base_dir, f"cleaned_{filename}")
        
        if os.path.exists(input_path):
            print(f"\n{'='*50}")
            print(f"处理文件: {filename}")
            print(f"{'='*50}")
            try:
                remove_duplicate_efsearch(input_path, output_path)
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")
        else:
            print(f"文件不存在: {input_path}")

if __name__ == "__main__":
    process_multiple_files() 