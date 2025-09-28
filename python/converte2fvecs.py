import h5py
import numpy as np
import os

RAW_DATA = "/home/wbh/cppwork/dco_benchmarks/DATA/hdf5"  
DATA_DIRECTORY = "/home/wbh/cppwork/dco_benchmarks/DATA"  
DATASETS = [
            # "glove-25-angular", 
            # "glove-50-angular", 
            # "glove-100-angular", 
            # "glove-200-angular", 
            # "sift-128-euclidean", 
            # "msong-420", 
            # "contriever-768", 
            # "gist-960-euclidean", 
            # "deep-image-96-angular", 
            "instructorxl-arxiv-768", 
            # "openai-1536-angular"
            ] 

def numpy_to_fvecs(numpy_array, output_file):
    """
    将numpy数组转换为fvecs格式并保存到文件
    
    Args:
        numpy_array: numpy数组，形状为(n, d)
        output_file: 输出文件路径
    """
    with open(output_file, 'wb') as f:
        for vector in numpy_array:
            # 写入维度（4字节整数）
            dim = len(vector)
            f.write(dim.to_bytes(4, byteorder='little'))
            # 写入向量数据（4字节浮点数）
            f.write(vector.astype(np.float32).tobytes())

def do_compute_gt(xb, xq, topk=100):
    nb, d = xb.shape
    index = faiss.IndexFlatL2(d)
    index.verbose = True
    index.add(xb)
    _, ids = index.search(x=xq, k=topk)
    return ids.astype('int32')

def convert_hdf5_to_fvecs(dataset):
    # 确保输出目录存在
    output_dir = os.path.join(DATA_DIRECTORY, "fvecs")
    os.makedirs(output_dir, exist_ok=True)
    
    hdf5_file_name = os.path.join(RAW_DATA, dataset + ".hdf5")
    hdf5_file = h5py.File(hdf5_file_name, "r")
    train_data = np.array(hdf5_file["train"], dtype=np.float32)
    test_data = np.array(hdf5_file["test"], dtype=np.float32)
        
    print(f"训练集形状: {train_data.shape}")
    print(f"测试集形状: {test_data.shape}")
        
    # 转换为fvecs格式并保存
    train_output = os.path.join(output_dir, dataset + '_base.fvecs')
    test_output = os.path.join(output_dir, dataset + '_query.fvecs')
    gt_output = os.path.join(output_dir, dataset + '_groundtruth.ivecs')
        
    print("正在转换训练集...")
    numpy_to_fvecs(train_data, train_output)
    print("正在转换测试集...")
    numpy_to_fvecs(test_data, test_output)
        
    print(f"转换完成！")
    print(f"训练集已保存到: {train_output}")
    print(f"测试集已保存到: {test_output}")


def main():
    # 设置输入输出路径
    for dataset in DATASETS:
        convert_hdf5_to_fvecs(dataset)

if __name__ == "__main__":
    main()