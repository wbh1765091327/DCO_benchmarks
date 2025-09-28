import pandas as pd
import os

def split_algorithm_data(input_file, output_dir):
    """
    将包含不同算法的CSV文件按算法分割成多个文件
    
    Args:
        input_file (str): 输入的CSV文件路径
        output_dir (str): 输出目录路径
    """
    # 读取CSV文件
    print(f"正在读取文件: {input_file}")
    df = pd.read_csv(input_file)
    
    # 获取所有唯一的算法名称
    algorithms = df['algorithm'].unique()
    print(f"发现 {len(algorithms)} 个算法: {algorithms}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 按算法分割数据
    for algorithm in algorithms:
        # 过滤出当前算法的数据
        algorithm_data = df[df['algorithm'] == algorithm]
        
        # 生成输出文件名（将算法名称中的特殊字符替换为下划线）
        safe_algorithm_name = algorithm.replace('-', '_').replace(' ', '_')
        output_filename = f"{safe_algorithm_name}.csv"
        output_path = os.path.join(output_dir, output_filename)
        
        # 保存到文件
        algorithm_data.to_csv(output_path, index=False)
        print(f"已保存 {len(algorithm_data)} 行数据到: {output_path}")
        
        # 显示该算法的数据集统计
        datasets = algorithm_data['dataset'].unique()
        print(f"  - 包含数据集: {datasets}")
        print(f"  - 数据集数量: {len(datasets)}")
        print()

def main():
    # 设置输入和输出路径
    input_file = r"E:\cppwork\dco_benchmarks\DATA\PDXnew\IVF_PDX_ADSAMPLING_nosimd.csv"
    output_dir = r"E:\cppwork\dco_benchmarks\DATA\PDXnew\splitnosimd"
    
    # 执行分割
    split_algorithm_data(input_file, output_dir)
    
    print("数据分割完成！")

if __name__ == "__main__":
    main()