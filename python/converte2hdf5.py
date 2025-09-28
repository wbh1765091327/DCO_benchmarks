import h5py
import numpy as np
import os
import struct

DATA_DIRECTORY = "/home/wbh/cppwork/dco_benchmarks/DATA"  
DATASETS = [
            # "glove-25-angular_100k", 
            # "glove-50-angular_100k", 
            # "glove-100-angular_100k", 
            # "glove-200-angular_100k", 
            # "glove-200-angular_1k",
            # "glove-200-angular_10k",
            # "instructorxl-arxiv-768_100k",
            "instructorxl-arxiv-768_1k",
            "instructorxl-arxiv-768_10k",
            "instructorxl-arxiv-768_1000k",
            ] 

def fvecs_to_numpy(fvecs_file):
    """
    读取fvecs文件并转换为numpy数组
    
    Args:
        fvecs_file: fvecs文件路径
        
    Returns:
        numpy数组，形状为(n, d)
    """
    vectors = []
    
    with open(fvecs_file, 'rb') as f:
        while True:
            # 读取维度（4字节整数）
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
                
            dim = struct.unpack('<I', dim_bytes)[0]  # little-endian unsigned int
            
            # 读取向量数据
            vector_bytes = f.read(dim * 4)  # 4字节浮点数
            if len(vector_bytes) != dim * 4:
                break
                
            vector = struct.unpack('<' + 'f' * dim, vector_bytes)
            vectors.append(vector)
    
    return np.array(vectors, dtype=np.float32)

def convert_fvecs_to_hdf5(dataset):
    """
    将base.fvecs和query.fvecs合并为一个HDF5文件
    
    Args:
        dataset: 数据集名称
    """
    # 确保输出目录存在
    output_dir = os.path.join(DATA_DIRECTORY, "hdf5")
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建文件路径
    fvecs_dir = os.path.join(DATA_DIRECTORY, "fvecs")
    data_dir = os.path.join(fvecs_dir, dataset)
    base_file = os.path.join(data_dir, dataset + '_base.fvecs')
    query_file = os.path.join(data_dir, dataset + '_query.fvecs')
    
    # 检查文件是否存在
    if not os.path.exists(base_file):
        print(f"错误: 找不到文件 {base_file}")
        return
    if not os.path.exists(query_file):
        print(f"错误: 找不到文件 {query_file}")
        return
    
    print(f"正在处理数据集: {dataset}")
    
    # 读取fvecs文件
    print("正在读取base.fvecs...")
    base_data = fvecs_to_numpy(base_file)
    print(f"Base数据形状: {base_data.shape}")
    
    print("正在读取query.fvecs...")
    query_data = fvecs_to_numpy(query_file)
    print(f"Query数据形状: {query_data.shape}")
    
    # 创建HDF5文件
    output_file = os.path.join(output_dir, dataset + ".hdf5")
    
    with h5py.File(output_file, 'w') as f:
        # 创建数据集
        f.create_dataset('train', data=base_data, compression='gzip', compression_opts=9)
        f.create_dataset('test', data=query_data, compression='gzip', compression_opts=9)
        
    
    print(f"转换完成！")
    print(f"HDF5文件已保存到: {output_file}")
    print(f"训练集: {base_data.shape[0]} 个向量")
    print(f"测试集: {query_data.shape[0]} 个向量")
    print(f"维度: {base_data.shape[1]}")


def main():
    # 基本转换
    print("=== 基本转换 ===")
    for dataset in DATASETS:
        convert_fvecs_to_hdf5(dataset)

if __name__ == "__main__":
    main()