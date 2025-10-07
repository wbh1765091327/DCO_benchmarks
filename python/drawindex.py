import matplotlib.pyplot as plt
import numpy as np

# --- 1. 准备数据和配置 ---

# X轴类别，两个图共享
datasets = ['Contriever', 'DEEP', 'GIST', 'GloVe', 'InstructorXL', 'MSong', 'OpenAI', 'SIFT']
# datasets = ['CTV', 'DEEP', 'GIST', 'GloVe', 'IXL', 'MSong', 'OpenAI', 'SIFT']
n_datasets = len(datasets)

# --- 数据和配置 for 左图 (HNSW Index Pre-Processing Time) ---
methods_time = ['ADS', "DADE", r"DDC$_{res}$", r"DDC$_{pca}$", r"DDC$_{opq}$", "RaBitQ", 'Finger','Flash']
data_time = {
    # 根据左图估算数据
    'ADS':       [14, 21, 22, 4, 59, 6, 25, 2],
    'DADE':       [159, 70, 194, 28, 260, 69, 325, 21],
    r"DDC$_{res}$":    [73, 85, 85, 16, 207, 33, 220, 13],
    r"DDC$_{pca}$":       [465, 164, 582, 157, 666, 345, 1050, 106],
    r"DDC$_{opq}$": [694, 148, 957, 93, 850, 311, 2702, 53],
    'RaBitQ': [19, 27, 24, 7, 42, 10, 39, 2],
    'Finger':       [331, 1609, 498, 121, 1009, 289, 872, 205],
    'Flash': [7, 13, 8, 4, 11, 4, 11, 3],
}

# --- 数据和配置 for 右图 (IVF Index Pre-Processing Time) ---
methods_space = ['ADS', "DADE", r"DDC$_{res}$", r"DDC$_{pca}$", r"DDC$_{opq}$","RaBitQ","Tribase","PDX",'SuCo']
data_space = {
    'ADS':       [14, 21, 22, 4, 59, 6, 25, 2],
    'DADE':       [159, 70, 194, 28, 260, 69, 325, 21],
    r"DDC$_{res}$":    [73, 85, 85, 16, 207, 33, 220, 13],
    r"DDC$_{pca}$":       [1059, 307, 1090, 346, 1748, 548, 1639, 153],
    r"DDC$_{opq}$": [1064, 475, 1311, 321, 1758, 433, 3031, 95],
    'RaBitQ': [19, 27, 24, 7, 42, 10, 39, 2] ,
    'Tribase': [78, 309, 115, 45, 235, 46, 171, 29],
    'PDX': [10, 13, 12, 4, 23, 6, 22, 2],
    'SuCo': [12, 1, 14, 3, 11, 6, 28, 2],
}

# --- 2. 定义样式 (覆盖所有方法) ---
# 使用不同颜色来区分不同方法
# col = ['#FF0000', '#0000FF', '#00AA00', '#FF8000', '#8000FF', '#FF1493', '#008B8B', '#B8860B', '#4B0082', '#228B22', '#FF4500','#FFD700']
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
# plt.savefig(f'E:/cppwork/dco_benchmarks/DATA/figure/ALL/HNSW/hnsw_index_volume.pdf', dpi=400, bbox_inches='tight')    

# 分别保存两张子图
# 保存左图 (HNSW)
fig_left = plt.figure(figsize=(10, 8))
ax_left = fig_left.add_subplot(111)
draw_chart(ax_left, data_time, methods_time, '(1) HNSW Pre-Processing Time', 'Time Consumption (Sec)',4)
plt.tight_layout(rect=[0, 0.2, 1, 0.85])
plt.savefig(f'E:/cppwork/dco_benchmarks/DATA/figure/ALL/HNSW/hnsw_preprocessing_time.pdf', dpi=400, bbox_inches='tight',format='pdf')
plt.close(fig_left)

# 保存右图 (IVF)
fig_right = plt.figure(figsize=(10, 8))
ax_right = fig_right.add_subplot(111)
draw_chart(ax_right, data_space, methods_space, '(2) IVF Pre-Processing Time', 'Time Consumption (Sec)',5)
plt.tight_layout(rect=[0, 0.2, 1, 0.85])
plt.savefig(f'E:/cppwork/dco_benchmarks/DATA/figure/ALL/HNSW/ivf_preprocessing_time.pdf', dpi=400, bbox_inches='tight',format='pdf')
plt.close(fig_right)

plt.show()