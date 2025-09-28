import pandas as pd
import os
import glob

def consolidate_tribase_nosimd_files(input_dir="E:/cppwork/dco_benchmarks/DATA/Tribase/tril2单线程SIMD", 
                                    output_file="E:/cppwork/dco_benchmarks/DATA/Tribase/IVF_Tril2_simd_new.csv"):
    """
    整合多线程非SIMD目录下的所有CSV文件为一个CSV文件
    
    Args:
        input_dir (str): 输入目录路径
        output_file (str): 输出文件路径
    
    Returns:
        pd.DataFrame: 整合后的数据框
    """
    print(f"正在处理目录: {input_dir}")
    
    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    if not csv_files:
        print("未找到CSV文件")
        return None
    
    all_dataframes = []
    
    # 处理每个CSV文件
    for csv_file in csv_files:
        print(f"处理文件: {os.path.basename(csv_file)}")
        
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file)
            
            # 从文件名提取数据集名称
            filename = os.path.basename(csv_file)
            # 移除.csv后缀和-nosimd后缀(如果存在)
            dataset_name = filename.replace('.csv', '').replace('-nosimd', '')
            
            # 如果CSV中的dataset列不是当前数据集名称，则更新它
            if 'dataset' in df.columns:
                # 检查是否数据集名称一致
                unique_datasets = df['dataset'].unique()
                if len(unique_datasets) == 1 and unique_datasets[0] != dataset_name:
                    print(f"  更新数据集名称: {unique_datasets[0]} -> {dataset_name}")
                    df['dataset'] = dataset_name
            else:
                # 如果没有dataset列，添加一个
                df['dataset'] = dataset_name
                print(f"  添加数据集列: {dataset_name}")
            
            all_dataframes.append(df)
            print(f"  处理完成: {len(df)} 行数据")
            
        except Exception as e:
            print(f"处理文件 {csv_file} 时出错: {e}")
    
    if not all_dataframes:
        print("没有成功处理任何文件")
        return None
    
    # 合并所有数据框
    print("\n正在合并所有数据框...")
    consolidated_df = pd.concat(all_dataframes, ignore_index=True)
    
    # 按数据集名称和nprobe排序
    if 'nprobe' in consolidated_df.columns:
        consolidated_df = consolidated_df.sort_values(['dataset', 'nprobe'])
    else:
        consolidated_df = consolidated_df.sort_values(['dataset'])
    
    # 重置索引
    consolidated_df = consolidated_df.reset_index(drop=True)
    
    print(f"合并完成! 总共 {len(consolidated_df)} 行数据")
    print(f"包含的数据集: {sorted(consolidated_df['dataset'].unique().tolist())}")
    
    # 保存到文件
    consolidated_df.to_csv(output_file, index=False)
    print(f"已保存到: {output_file}")
    
    return consolidated_df

if __name__ == "__main__":
    # 整合文件
    result_df = consolidate_tribase_nosimd_files()
    
    if result_df is not None:
        print("\n数据预览:")
        print(result_df.head())
        print(f"\n列名: {list(result_df.columns)}")
        print(f"数据形状: {result_df.shape}")
        print(f"\n各数据集的数据量:")
        print(result_df['dataset'].value_counts().sort_index()) 