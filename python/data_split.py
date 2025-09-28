from utils import fvecs_read
from utils import fvecs_write, ivecs_read, ivecs_write
import os
import numpy as np
import faiss

source = '/home/wbh/cppwork/dco_benchmarks/DATA/fvecs'
datasets = [
            'glove-25-angular',
            'glove-100-angular',
            'glove-50-angular', 
            'glove-200-angular',
            # 'sift-128-euclidean',
            # 'msong-420', 
            # 'contriever-768', 
            # 'gist-960-euclidean', 
            # 'deep-image-96-angular', 
            # 'instructorxl-arxiv-768', 
            # 'openai-1536-angular'
            ]

def do_compute_gt(xb, xq, topk=100):
    nb, d = xb.shape
    index = faiss.IndexFlatL2(d)
    index.verbose = True
    index.add(xb)
    _, ids = index.search(x=xq, k=topk)
    return ids.astype('int32')


if __name__ == "__main__":
    for dataset in datasets:
        print(f'current dataset: {dataset}')
        base_path = os.path.join(source, f'{dataset}_base.fvecs')
        query_path = os.path.join(source, f'{dataset}_query.fvecs')
        origin_data = fvecs_read(base_path)
        query_data = fvecs_read(query_path)
        np.random.shuffle(origin_data)

        origin_num = origin_data.shape[0]
        learn_num = int(1e4) - int(1e3)
        base_num = int(1e4)

        #learn_data is for model learning
        learn_data = origin_data[:learn_data]
        base_data = origin_data[learn_num:learn_num+base_num]

        gt = do_compute_gt(base_data, query_data, topk=10)
        learn_gt = do_compute_gt(base_data, learn_data, topk=10)

        save_path = os.path.join(source, f'{dataset}_1k')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_base_path = os.path.join(save_path, f'{dataset}_1k_base.fvecs')
        save_learn_path = os.path.join(save_path, f'{dataset}_1k_learn.fvecs')
        save_ground_path = os.path.join(save_path, f'{dataset}_1k_groundtruth.ivecs')
        save_learn_ground_path = os.path.join(save_path, f'{dataset}_1k_learn_groundtruth.ivecs')

        fvecs_write(save_base_path, base_data)
        fvecs_write(save_learn_path, learn_data)
        ivecs_write(save_ground_path, gt)
        ivecs_write(save_learn_ground_path, learn_gt)