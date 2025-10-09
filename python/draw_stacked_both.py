import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# --- 1. 准备数据和配置 ---
# 基础字体
mpl.rcParams['font.family'] = 'Arial Unicode MS'

# 让 mathtext 使用 Arial Unicode MS
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Arial Unicode MS'
mpl.rcParams['mathtext.it'] = 'Arial Unicode MS'
mpl.rcParams['mathtext.bf'] = 'Arial Unicode MS'

# X轴类别
datasets = ['Contriever', 'DEEP', 'GIST', 'GloVe', 'InstructorXL', 'MSong', 'OpenAI', 'SIFT']
n_datasets = len(datasets)

# ========== HNSW 数据 ==========
methods_hnsw = ['ADS', "DADE", r"DDC$_{res}$", r"DDC$_{pca}$", r"DDC$_{opq}$", "RaBitQ", 'Finger', 'Flash']

# HNSW Preprocessing Time 数据（来自 drawindex.py 的 data_time）
hnsw_preprocessing = {
    'ADS':       [14, 21, 22, 4, 59, 6, 25, 2],
    'DADE':       [159, 70, 194, 28, 260, 69, 325, 21],
    r"DDC$_{res}$":    [73, 85, 85, 16, 207, 33, 220, 13],
    r"DDC$_{pca}$":       [465, 164, 582, 157, 666, 345, 1050, 106],
    r"DDC$_{opq}$": [694, 148, 957, 93, 850, 311, 2702, 53],
    'RaBitQ': [19, 27, 24, 7, 42, 10, 39, 2],
    'Finger':       [331, 1609, 498, 121, 1009, 289, 872, 205],
    'Flash': [7, 13, 8, 4, 11, 4, 11, 3],
}

# HNSW Total Time 数据（来自 draw_index_sum.py 的 data_time）
hnsw_total = {
    'ADS':      [273, 589, 363, 73, 782, 105, 691, 47],
    'DADE':     [418, 638, 535, 97, 983, 168, 991, 66],
    r"DDC$_{res}$":  [332, 653, 426, 85, 930, 132, 886, 58],
    r"DDC$_{pca}$":  [724, 732, 923, 226, 1389, 444, 1716, 151],
    r"DDC$_{opq}$":  [953, 716, 1298, 162, 1573, 410, 3368, 98],
    'RaBitQ':   [317, 809, 393, 143, 743, 149, 528, 62],
    'Finger':   [590, 2177, 839, 190, 1732, 388, 1538, 250],
    'Flash':    [38, 548, 48, 44, 104, 48, 47, 34],
}

# ========== IVF 数据 ==========
methods_ivf = ['ADS', "DADE", r"DDC$_{res}$", r"DDC$_{pca}$", r"DDC$_{opq}$", "RaBitQ", "Tribase", "PDX", 'SuCo']

# IVF Preprocessing Time 数据（来自 drawindex.py 的 data_space）
ivf_preprocessing = {
    'ADS':       [14, 21, 22, 4, 59, 6, 25, 2],
    'DADE':       [159, 70, 194, 28, 260, 69, 325, 21],
    r"DDC$_{res}$":    [73, 85, 85, 16, 207, 33, 220, 13],
    r"DDC$_{pca}$":       [1059, 307, 1090, 346, 1748, 548, 1639, 153],
    r"DDC$_{opq}$": [1064, 475, 1311, 321, 1758, 433, 3031, 95],
    'RaBitQ': [19, 27, 24, 7, 42, 10, 39, 2],
    'Tribase': [78, 309, 115, 45, 235, 46, 171, 29],
    'PDX': [10, 13, 12, 4, 23, 6, 22, 2],
    'SuCo': [12, 1, 14, 3, 11, 6, 28, 2],
}

# IVF Total Time 数据（来自 draw_index_sum.py 的 data_space）
ivf_total = {
    'ADS':      [211, 951, 274, 65, 539, 113, 418, 37],
    'DADE':     [356, 1000, 446, 89, 740, 176, 718, 56],
    r"DDC$_{res}$":  [270, 1015, 337, 77, 687, 140, 613, 48],
    r"DDC$_{pca}$":  [1256, 1237, 1342, 407, 2228, 655, 2032, 188],
    r"DDC$_{opq}$":  [1261, 1405, 1563, 382, 2238, 540, 3424, 130],
    'RaBitQ':   [216, 957, 276, 68, 522, 117, 432, 37],
    'Tribase':  [275, 1239, 367, 106, 715, 153, 564, 64],
    'PDX':      [207, 943, 264, 65, 503, 113, 415, 37],
    'SuCo':     [17, 19, 19, 7, 19, 13, 36, 4],
}

# --- 2. 定义样式 ---
styles = {
    'ADS':                  {'color': '#484F98', 'edgecolor': 'black'},   # 深蓝
    'DADE':                 {'color': '#5DADF2', 'edgecolor': 'black'},   # 浅蓝
    r"DDC$_{res}$":         {'color': '#91DFFF', 'edgecolor': 'black'},   # 天蓝
    r"DDC$_{pca}$":         {'color': '#52607E', 'edgecolor': 'black'},   # 蓝灰
    r"DDC$_{opq}$":         {'color': '#A46DAD', 'edgecolor': 'black'},   # 紫色
    'RaBitQ':               {'color': '#8A7067', 'edgecolor': 'black'},   # 灰褐
    'Tribase':              {'color': '#F6DA65', 'edgecolor': 'black'},   # 浅黄
    'PDX':                  {'color': '#F5B041', 'edgecolor': 'black'},   # 金黄
    'Finger':               {'color': '#D5695D', 'edgecolor': 'black'},   # 红棕
    'Flash':                {'color': '#FFBCAB', 'edgecolor': 'black'},   # 浅粉
    'SuCo':                 {'color': '#FFDFFF', 'edgecolor': 'black'},   # 浅粉紫
}

# --- 3. 绘制堆叠柱状图函数 ---
def draw_stacked_chart(ax, data_preprocessing, data_total, methods, ylabel, ncol):
    """绘制堆叠柱状图"""
    # 计算 Indexing Time（总时间 - Preprocessing Time）
    data_indexing = {}
    for method in methods:
        data_indexing[method] = [
            data_total[method][i] - data_preprocessing[method][i] 
            for i in range(n_datasets)
        ]
    
    n_methods = len(methods)
    bar_width = 0.8 / n_methods
    index = np.arange(n_datasets)
    
    bars_list_bottom = []
    bars_list_top = []
    positions_list = []

    for i, method in enumerate(methods):
        positions = index + (i - n_methods / 2 + 0.5) * bar_width
        values_bottom = data_preprocessing[method]
        values_top = data_indexing[method]
        style = styles[method]
        
        # 绘制下半部分（Preprocessing）- 深色
        bars_bottom = ax.bar(
            positions, values_bottom, width=bar_width,
            edgecolor='black', color=style['color'], linewidth=1,
            label=method
        )
        
        # 绘制上半部分（Indexing）- 浅色半透明
        bars_top = ax.bar(
            positions, values_top, width=bar_width, bottom=values_bottom,
            edgecolor='black', color=style['color'], linewidth=1,
            alpha=0.8
        )
        
        bars_list_bottom.append(bars_bottom)
        bars_list_top.append(bars_top)
        positions_list.append(positions)

    # 在每个数据集下找到总时间最低的柱子并添加红色五角星
    for dataset_idx in range(n_datasets):
        dataset_total_values = [data_total[method][dataset_idx] for method in methods]
        min_total_value = min(dataset_total_values)
        min_method_idx = dataset_total_values.index(min_total_value)
        
        bar_x = positions_list[min_method_idx][dataset_idx]
        bar_height = min_total_value
        
        # 在最低总时间柱子上添加红色五角星
        ax.plot(bar_x, bar_height * 1.5, 'r*', markersize=7, markeredgecolor='red', 
                markerfacecolor='red', markeredgewidth=0.5)

    # 美化坐标轴
    ax.set_yscale('log')
    ax.set_ylim(10**0, 10**4)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xticks(index)
    ax.set_xticklabels(datasets, rotation=0, ha='center')
    ax.tick_params(axis='both', which='major', labelsize=13)
    
    # 图例
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=ncol, 
              frameon=False, columnspacing=1, fontsize=16)
    
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# --- 4. 创建并保存HNSW堆叠图 ---
print("正在生成 HNSW 堆叠柱状图...")
fig_hnsw = plt.figure(figsize=(10, 8))
ax_hnsw = fig_hnsw.add_subplot(111)

draw_stacked_chart(ax_hnsw, hnsw_preprocessing, hnsw_total, methods_hnsw, 'Time (s)', 4)

plt.tight_layout(rect=[0, 0.2, 1, 0.85])
plt.savefig(f'E:/cppwork/dco_benchmarks/DATA/figure/ALL/HNSW/hnsw_stacked_time.pdf', 
            dpi=400, bbox_inches='tight', format='pdf')
plt.close(fig_hnsw)
print("✓ HNSW堆叠图已保存到: E:/cppwork/dco_benchmarks/DATA/figure/ALL/HNSW/hnsw_stacked_time.pdf")

# --- 5. 创建并保存IVF堆叠图 ---
print("\n正在生成 IVF 堆叠柱状图...")
fig_ivf = plt.figure(figsize=(10, 8))
ax_ivf = fig_ivf.add_subplot(111)

draw_stacked_chart(ax_ivf, ivf_preprocessing, ivf_total, methods_ivf, 'Time (s)', 5)

plt.tight_layout(rect=[0, 0.2, 1, 0.85])
plt.savefig(f'E:/cppwork/dco_benchmarks/DATA/figure/ALL/HNSW/ivf_stacked_time.pdf', 
            dpi=400, bbox_inches='tight', format='pdf')
plt.close(fig_ivf)
print("✓ IVF堆叠图已保存到: E:/cppwork/dco_benchmarks/DATA/figure/ALL/HNSW/ivf_stacked_time.pdf")

print("\n=== 完成！两张堆叠柱状图已生成 ===")

# # 可选：显示图表
# fig_hnsw = plt.figure(figsize=(10, 8))
# ax_hnsw = fig_hnsw.add_subplot(111)
# draw_stacked_chart(ax_hnsw, hnsw_preprocessing, hnsw_total, methods_hnsw, 'Time (s)', 4)
# plt.tight_layout(rect=[0, 0.05, 1, 0.85])
# plt.title('HNSW Stacked Time', fontsize=18, pad=20)

# fig_ivf = plt.figure(figsize=(10, 8))
# ax_ivf = fig_ivf.add_subplot(111)
# draw_stacked_chart(ax_ivf, ivf_preprocessing, ivf_total, methods_ivf, 'Time (s)', 5)
# plt.tight_layout(rect=[0, 0.05, 1, 0.85])
# plt.title('IVF Stacked Time', fontsize=18, pad=20)

# plt.show()

