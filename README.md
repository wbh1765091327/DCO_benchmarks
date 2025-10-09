# Distance Comparison Operations Optimization in Approximate Nearest Neighbor Search: A Survey and Experimental Evaluation

## Introduction

This work investigated and benchmarked mainstream DCO optimization methods. We divided existing methods into projection pruning based and quantization based techniques, and reviewed representative methods in each category. Extensive experiments were conducted on eight datasets, covering different index structures and implementations, evaluating key metrics such as time accuracy trade-offs and preprocessing overhead, while examining their scalability in terms of data dimensions and scale. Based on our findings, we provide performance characteristics to guide method selection and point out promising directions for future research.

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

We also provide a dockerfile based on Ubuntu22.04 with all the dependencies installed. All the following experiments need to set the base or source path in the py and sh files according to their own environment, otherwise an error will occur when running.

## Setup dataset

Firstly, download the data to the/benchmarks/DATA/hdv5 directory, convert the hdf5 format to query and base in fvecs format, and then decide whether to execute data_stplit. py based on the experimental requirements (complete dataset or partial dataset).

```sh
cd benchmarks
pip install -r ./python/requirements.txt
python ./benchmarks/python/converte2fvecs.py
python ./benchmarks/python/data_split.py
```

## How to Run

### Res-Info

We first test Baseline, ADSamping, DDC res, DDC pca, DDC opq, Finger
Firstly, place the base, query, and gt files of different datasets after data processing into the Res Transfer/DATA/${dataset} directory. Then compile.

Preprocessing and construction of indexes for Baseline and ADSampling
```bash
./make_dir.sh
cd ../script/
./pre_compute.sh
./index_ivf.sh
./index_hnsw.sh
```
Index preprocessing and construction for DDC res, DDC pca, and DDC opq
```bash
cd ../script/
./index_pca.sh
./index_opq.sh
./linear.sh
```

Finger's index preprocessing and construction
```bash
python3 finger.py
```

Modify compilation conditions and compile
```bash
cd build && make build
./search_ivf.sh
./search_hnsw.sh
./search_ivf_avx512.sh
./search_hnsw_avx512.sh
./search_hnsw_finger.sh
```
### DADE

Firstly, place the base, query, and gt files of different datasets after data processing into the DADE/data/${dataset} directory. Then compile, and also modify the add_definitions condition in DADE/CMakeLists.txt.
```bash
./index_ivf.sh
./index_hnsw.sh
./search_ivf.sh
./search_ivf_simd.sh
./search_hnsw.sh
./search_hnsw_simd.sh
```

### PDX

First, place the HDF5 file into the PDX/benchmarks/download directory. Firstly, modify the DATASETS and DIMENSIONALITIES in set_detting.py, as well as the DATASETS in PDX/include/tilt/benchmark_utils.hpp.
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

Firstly, place the base, query, and gt files of different datasets after data processing into the RaBitQ Library/data/${dataset} directory. Then compile and execute Example 2. sh.
```bash
cd build && make build
cd ..
chmod +x example2.sh && ./example2.sh
```
### Flash

Firstly, place the base, query, and gt files of different datasets after data processing into the HNSW Flash/data/${dataset} directory.
```bash
make build
cd ./bin && ./main -k 10 -s 200 -v 16 -p 16 ${dataset} flash
```

### Tribase

Firstly, place the base and query files of different datasets after data processing into the Tribase/benchmarks/${dataset}/origin directory, and modify the dataset array in query.cpp.
```bash
cd build && make build
./build/bin/query  --opt_levels  OPT_ALL --nprobes 2 4 6 8 10 12 14 16 18 20 22 24 26 28 32 40 48 56 64 80 96 112 128 144 160 192 224 256 512 --cache  --verbose
```

### SuCo

Firstly, place the base, query, and gt files of different datasets after data processing into the SuCo/data/${dataset} directory.
```bash
make
cd ./script && ./run.sh
```
