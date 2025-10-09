import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
# --- 1. 准备数据和配置 ---

# X轴类别，两个图共享
datasets = ['Contriever', 'DEEP', 'GIST', 'GloVe', 'InstructorXL', 'MSong', 'OpenAI', 'SIFT']
n_datasets = len(datasets)

# --- 数据和配置 for 左图 (HNSW Index Pre-Processing Time) ---
methods_time = ['ADS', "DADE", r"DDC$_{res}$", r"DDC$_{pca}$", r"DDC$_{opq}$", "RaBitQ", 'Finger','Flash']

# 基础字体
mpl.rcParams['font.family'] = 'Arial Unicode MS'

# 让 mathtext 使用 Arial Unicode MS
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Arial Unicode MS'
mpl.rcParams['mathtext.it'] = 'Arial Unicode MS'
mpl.rcParams['mathtext.bf'] = 'Arial Unicode MS'

data_time = {
    # 根据左图估算数据
    'ADS':      [273, 589, 363, 73, 782, 105, 691, 47],
    'DADE':     [418, 638, 535, 97, 983, 168, 991, 66],
    r"DDC$_{res}$":  [332, 653, 426, 85, 930, 132, 886, 58],
    r"DDC$_{pca}$":  [724, 732, 923, 226, 1389, 444, 1716, 151],
    r"DDC$_{opq}$":  [953, 716, 1298, 162, 1573, 410, 3368, 98],
    'RaBitQ':   [317, 809, 393, 143, 743, 149, 528, 62],
    'Finger':   [590, 2177, 839, 190, 1732, 388, 1538, 250],
    'Flash':    [38, 548, 48, 44, 104, 48, 47, 34],
}

# --- 数据和配置 for 右图 (IVF Index Pre-Processing Time) ---
methods_space = ['ADS', "DADE", r"DDC$_{res}$", r"DDC$_{pca}$", r"DDC$_{opq}$","RaBitQ","Tribase","PDX",'SuCo']
data_space = {
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

# --- 2. 定义样式 (覆盖所有方法) ---
# 使用不同颜色来区分不同方法
# col = ['#FF0000', '#0000FF', '#00AA00', '#FF8000', '#8000FF', '#FF1493', '#008B8B', '#B8860B', '#4B0082', '#228B22', '#FF4500','#FFD700']
# col = ['#E76254', '#EF8A47', '#F7AA58', '#FFDC6F', '#FFAAE7', '#AADED8', '#7290D5', '#52708D', '#376795', '#1E466E', '#FF4500']
# col = ['#001219', '#006073', '#099693', '#91D3C0', '#EBD7A5', '#EE9B9B', '#CC6602', '#BC3E03', '#AE2012', '#9B2226', '#FF4500']
# col = [
#     '#D5695D',  # R:213 G:105 B:093
#     '#F5B041',  # R:245 G:176 B:065
#     '#F6DA65',  # R:246 G:218 B:101
#     '#52607E',  # R:082 G:190 B:128
#     '#91DFFF',  # R:145 G:223 B:208
#     '#5DADF2',  # R:093 G:173 B:226
#     '#A46DAD',  # R:164 G:105 B:189
#     '#8A7067',  # R:138 G:112 B:103
#     '#FFBCAB',  # R:255 G:188 B:167
#     '#484F98',  # R:072 G:079 B:152
#     '#FFDFFF',  # R:255 G:255 B:133
# ]
styles = {
    'ADS':                  {'hatch': '', 'color': '#484F98', 'edgecolor': 'black'},   # 深蓝
    'DADE':                 {'hatch': '', 'color': '#5DADF2', 'edgecolor': 'black'},   # 浅蓝
    r"DDC$_{res}$":         {'hatch': '', 'color': '#91DFFF', 'edgecolor': 'black'},   # 天蓝
    r"DDC$_{pca}$":         {'hatch': '', 'color': '#52607E', 'edgecolor': 'black'},   # 蓝灰
    r"DDC$_{opq}$":         {'hatch': '', 'color': '#A46DAD', 'edgecolor': 'black'},   # 紫色
    'RaBitQ':               {'hatch': '', 'color': '#8A7067', 'edgecolor': 'black'},   # 灰褐
    'Tribase':              {'hatch': '', 'color': '#F6DA65', 'edgecolor': 'black'},   # 浅黄
    'PDX':                  {'hatch': '', 'color': '#F5B041', 'edgecolor': 'black'},   # 金黄
    'Finger':               {'hatch': '', 'color': '#D5695D', 'edgecolor': 'black'},   # 红棕
    'Flash':                {'hatch': '', 'color': '#FFBCAB', 'edgecolor': 'black'},   # 浅粉
    'SuCo':                 {'hatch': '', 'color': '#FFDFFF', 'edgecolor': 'black'},   # 浅粉紫
}

# --- 3. 封装绘图函数 ---
def draw_chart(ax, data, methods, title, ylabel,ncol):
    """在一个指定的坐标轴上绘制分组条形图"""
    n_methods = len(methods)
    bar_width = 0.8 / n_methods  # 调整条形宽度以适应组
    index = np.arange(n_datasets)
    
    # 存储所有条形对象，用于后续标记
    bars_list = []
    positions_list = []

    for i, method in enumerate(methods):
        positions = index + (i - n_methods / 2 + 0.5) * bar_width
        values = data[method]
        style = styles[method]
        bars = ax.bar(
            positions, values, width=bar_width, label=method,
            edgecolor='black', hatch=style['hatch'], color=style['color'], linewidth=1
        )
        bars_list.append(bars)
        positions_list.append(positions)

    # 在每个数据集下找到最低开销的柱子并添加红色五角星
    for dataset_idx in range(n_datasets):
        # 找到该数据集下所有方法的值
        dataset_values = [data[method][dataset_idx] for method in methods]
        min_value = min(dataset_values)
        min_method_idx = dataset_values.index(min_value)
        
        # 获取最低开销柱子的位置
        bar_x = positions_list[min_method_idx][dataset_idx]
        bar_height = min_value
        
        # 在最低开销柱子上添加红色五角星
        ax.plot(bar_x, bar_height * 1.5, 'r*', markersize=7, markeredgecolor='red', 
                markerfacecolor='red', markeredgewidth=0.5)

    # 美化坐标轴
    ax.set_yscale('log')
    ax.set_ylim(10**-1, 10**4)
    ax.set_ylabel(ylabel, fontsize=16)
    # 将标题放在图片下面
    # ax.text(0.5, -0.25, title, transform=ax.transAxes, fontsize=14, ha='center', va='top')
    ax.set_xticks(index)
    ax.set_xticklabels(datasets, rotation=0, ha='center')
    ax.tick_params(axis='both', which='major', labelsize=13)
    
    # 图例
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=ncol, frameon=False,columnspacing=1, fontsize=16)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# --- 4. 创建并绘制图表 ---
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# # 绘制左图
# draw_chart(ax1, data_time, methods_time, '(1) HNSW Pre-Processing Time', 'Time Consumption (Sec)')

# # 绘制右图
# draw_chart(ax2, data_space, methods_space, '(2) IVF Pre-Processing Time', 'Time Consumption (Sec)')

# # 调整整体布局，防止重叠，为下面的标题留出空间
# plt.tight_layout(rect=[0, 0.2, 1, 0.85]) # rect=[left, bottom, right, top] 为标题和图例留出空间

# # 保存完整的两子图组合
# plt.savefig(f'E:/cppwork/dco_benchmarks/DATA/figure/ALL/HNSW/hnsw_index_volume-sum.pdf', dpi=400, bbox_inches='tight')    

# 分别保存两张子图
# 保存左图 (HNSW)
fig_left = plt.figure(figsize=(10, 8))
ax_left = fig_left.add_subplot(111)
draw_chart(ax_left, data_time, methods_time, '(1) HNSW Pre-Processing Time', 'Time (s)',4)
plt.tight_layout(rect=[0, 0.2, 1, 0.85])
plt.savefig(f'E:/cppwork/dco_benchmarks/DATA/figure/ALL/HNSW/hnsw_preprocessing_time_sum.pdf', dpi=400, bbox_inches='tight',format='pdf')
plt.close(fig_left)

# 保存右图 (IVF)
fig_right = plt.figure(figsize=(10, 8))
ax_right = fig_right.add_subplot(111)
draw_chart(ax_right, data_space, methods_space, '(2) IVF Pre-Processing Time', 'Time (s)',5)
plt.tight_layout(rect=[0, 0.2, 1, 0.85])
plt.savefig(f'E:/cppwork/dco_benchmarks/DATA/figure/ALL/HNSW/ivf_preprocessing_time_sum.pdf', dpi=400, bbox_inches='tight',format='pdf')
plt.close(fig_right)

plt.show()