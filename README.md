# Distance Comparison Operations Optimization in Approximate Nearest Neighbor Search: A Survey and Experimental Evaluation

## Introduction


## Dataset

We have prepared some real-world datasets for testing in benchmarks folder. You can download the large datasets from the following [links](https://drive.google.com/drive/folders/1f76UCrU52N2wToGMFg9ir1MY8ZocrN34):


| Dataset     | Dimension | Number of Data Items | Number of Queries | Type   | Distribution |
|-------------|-----------|-----------------------|-------------------|--------|--------------|
| GloVe       | 25        | 1,183,514             | 10,000            | Text   | Normal       |
| GloVe       | 50        | 1,183,514             | 10,000            | Text   | Normal       |
| DEEP        | 96        | 9,990,000             | 10,000            | Image  | Normal       |
| GloVe       | 100       | 1,183,514             | 10,000            | Text   | Normal       |
| SIFT        | 128       | 1,000,000             | 10,000            | Image  | Skewed       |
| GloVe       | 200       | 1,183,514             | 10,000            | Text   | Normal       |
| MSong       | 420       | 983,185               | 1,000             | Audio  | Skewed       |
| Contriever  | 768       | 990,000               | 10,000            | Text   | Normal       |
| Instructorxl| 768       | 2,253,000             | 1,000             | Text   | Normal       |
| GIST        | 960       | 1,000,000             | 1,000             | Image  | Skewed       |
| OpenAI      | 1536      | 999,000               | 1,000             | Text   | Skewed       |

### Dataset Format

You need to place the hdf5 data file in the dcoubenchmarks \ DATA \ hdf5 directory, or place the fvecs file in the dcoubenchmarks \ DATA \ vecs directory. You can do it through the converte2hdf5. py and converte2fvecs. py files in the dcoubenchmarks \ python directory


## Experimental Setup

Our server setup includes two Intel Xeon Gold 5318Y CPUs, each with 24 cores and 48 threads, totaling 96 CPU cores. The server boasts 2TB of memory and runs on CentOS Stream 8 operating system.

We also provide a dockerfile based on Ubuntu22.04 with all the dependencies installed. All the following experiments need to set the source path according to their own environment, otherwise an error will occur when running.

## Setup dataset
首先将数据下载到/benchmarks/DATA/hdf5目录下，将hdf5格式转化为fvecs格式的query和base，然后根据实验需求(完整数据集还是部分数据集)决定是否执行data_split.py。

```sh
cd benchmarks
pip install -r ./python/requirements.txt
python ./benchmarks/python/converte2fvecs.py
python ./benchmarks/python/data_split.py
```

## How to Run

### Res-Info
我们首先测试 Baseline , ADSampling , DDC-res , DDC-pca, DDC-opq, Finger
首先将数据处理后的不同dataset的base,query,gt三个文件放入到Res-Infer/DATA/${dataset}目录下。然后编译

Baseline和ADSampling的索引预处理和构建
```bash
./make_dir.sh
cd ../script/
./pre_compute.sh
./index_ivf.sh
./index_hnsw.sh
```
DDC-res,DDC-pca,DDC-opq的索引预处理和构建
```bash
cd ../script/
./index_pca.sh
./index_opq.sh
./linear.sh
```

Finger的索引预处理和构建
```bash
python3 finger.py
```

修改编译条件并编译
```bash
cd build && make build
./search_ivf.sh
./search_hnsw.sh
./search_ivf_avx512.sh
./search_hnsw_avx512.sh
./search_hnsw_finger.sh
```
### DADE

首先将数据处理后的不同dataset的base,query,gt三个文件放入到DADE/data/${dataset}目录下。然后编译,同样需要在DADE/CMakeLists.txt下修改add_definitions条件。
```bash
./index_ivf.sh
./index_hnsw.sh
./search_ivf.sh
./search_ivf_simd.sh
./search_hnsw.sh
./search_hnsw_simd.sh
```

### PDX

首先将hdf5文件放入到PDX/benchmarks/download目录下。首先修改set_settings.py中的DATASETS和DIMENSIONALITIES以及PDX/include/utils/benchmark_utils.hpp中过的DATASETS。
```bash
python3 ./benchmarks/python_scripts/setup_data.py
# GRAVITON4
cmake . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O3 -mcpu=neoverse-v2"
# GRAVITON3
cmake . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O3 -mcpu=neoverse-v1"
# Intel Sapphire Rapids (256 vectors are used if mprefer-vector-width is not specified)
cmake . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=sapphirerapids -mtune=sapphirerapids -mprefer-vector-width=512"
# ZEN4
cmake . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=znver4 -mtune=znver4"
# ZEN3
cmake . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=znver3 -mtune=znver3"
# 是否开启PDX_USE_EXPLICIT_SIMD标记
./benchmarks/benchmark_ivf.sh python3
```

### RabitQ

首先将数据处理后的不同dataset的base,query,gt三个文件放入到RaBitQ-Library/data/${dataset}目录下。然后编译并执行example2.sh。
```bash
cd build && make build
cd ..
chmod +x example2.sh && ./example2.sh
```
### Flash

首先将数据处理后的不同dataset的base,query,gt三个文件放入到HNSW-Flash/data/${dataset}目录下。。
```bash
make build
cd ./bin && ./main -k 10 -s 200 -v 16 -p 16 ${dataset} flash
```

### Tribase

首先将数据处理后的不同dataset的base,query两个个文件放入到Tribase/benchmarks/${dataset}/origin目录下,修改query.cpp中的dataset数组。
```bash
cd build && make build
./build/bin/query  --opt_levels  OPT_ALL --nprobes 2 4 6 8 10 12 14 16 18 20 22 24 26 28 32 40 48 56 64 80 96 112 128 144 160 192 224 256 512 --cache  --verbose
```

### SuCo

首先将数据处理后的不同dataset的base,query,gt三个文件放入到SuCo/data/${dataset}目录下。
```bash
make
cd ./script && ./run.sh
```
