import matplotlib.pyplot as plt
import numpy as np

# 1. 准备数据 (根据您的指正，排除了 IVF/HNSW 行)
# 列（X轴）是数据集
datasets = ['Contriever', 'DEEP', 'GIST', 'GloVe', 'Instructorxl', 'MSong', 'OpenAI', 'SIFT']
# 行（Y轴的系列）是不同的方法
methods = ['ADS', 'DADE', r"DDC$_{res}$", r"DDC$_{pca}$", r"DDC$_{opq}$", 'RaBitQ','Finger']

# 将所有百分比数据存入一个字典
data = {
    'ADS': [0.21, 0.28, 0.31, 0.14, 0.35, 0.41, 0.41, 0.46],
    'Finger':     [4.81, 13.16, 3.82, 1.39, 10.54, 9.25, 12.37, 7.24],
    'DADE':       [0.29, 0.0, 0.01, 4.66, 0, 0.01, 0.06, 0.02],
    r"DDC$_{res}$":    [2.51, 2.21, 0.34, 0.37, 0.55, 1.26, 0.63, 0.15],
    r"DDC$_{pca}$":  [0.32, 0, 0.05, 0.33, 0.08, 0.11, 0.12, 0.2],
    r"DDC$_{opq}$":  [1.89, 0.01, 0.09, 0.06, 0.05, 0.47, 0.28, 0],
    'RaBitQ':     [0.17, 0.147, 0.77, 0.46, 0.27, 0.57, 0.08, 0.43]
}

# 定义黑白样式 - 使用更规律、更美观的图案
styles = {
    'ADS':      {'hatch': '', 'color': '#0000FF', 'edgecolor': 'black'},
    'DADE':       {'hatch': '', 'color': '#00AA00', 'edgecolor': 'black'},
    r"DDC$_{res}$":       {'hatch': '', 'color': '#FF8000', 'edgecolor': 'black'},
    r"DDC$_{pca}$":       {'hatch': '', 'color': '#8000FF', 'edgecolor': 'black'},
    r"DDC$_{opq}$": {'hatch': '', 'color': '#FF1493', 'edgecolor': 'black'},
    'RaBitQ': {'hatch': '', 'color': '#008B8B', 'edgecolor': 'black'},
    'Finger':       {'hatch': '', 'color': '#228B22', 'edgecolor': 'black'},
}

# 2. 创建图表
# 创建一个标准的图表和坐标轴
fig, ax = plt.subplots(figsize=(15, 8))

# 计算条形的位置
n_methods = len(methods)
n_datasets = len(datasets)
# 减小条形宽度以适应更多的组
bar_width = 0.11 
index = np.arange(n_datasets) # 每组条形的中心位置

# 设置y轴上限
y_max = 3.0

# 3. 循环绘制条形图
for i, method in enumerate(methods):
    # 计算当前方法所有条形的位置
    positions = index + (i - n_methods / 2 + 0.5) * bar_width
    values = data[method]
    
    # 限制值不超过y轴上限
    capped_values = [min(val, y_max) for val in values]
    
    # 绘制柱子
    bars = ax.bar(
        positions, 
        capped_values, 
        width=bar_width, 
        label=method,
        color=styles[method]['color'],
        edgecolor=styles[method]['edgecolor'],
        hatch=styles[method]['hatch'],
        linewidth=0.5
    )
    
    # 在柱子上添加数值标签
    for j, (bar, original_val, capped_val) in enumerate(zip(bars, values, capped_values)):
        # 只有原始值超过上限的柱子才显示数值
        if original_val > y_max:
            # 在柱子顶部添加一个小的黑色标记表示截断
            ax.plot(bar.get_x() + bar.get_width()/2, y_max, 'k^', markersize=8, markeredgecolor='black', markerfacecolor='black')
            # 显示原始值
            ax.text(bar.get_x() + bar.get_width()/2, y_max + 0.1, 
                   f'{original_val:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

# 4. 美化图表
# 设置Y轴标签和上限
ax.set_ylabel('FPR@K (%)', fontsize=20)
ax.set_ylim(0, y_max)  # 设置y轴范围，不超过3.0

# # 设置X轴
# ax.set_xlabel('Dataset', fontsize=14)
ax.set_xticks(index)
ax.set_xticklabels(datasets, fontsize=16, rotation=45, ha='right')
ax.tick_params(axis='y', labelsize=16)

# 添加网格线，使数值更容易读取
ax.yaxis.grid(True, linestyle='--', alpha=0.7)

# 添加图例 - 放在图外部，去掉边框和标题，两行居中显示（第一行4个，第二行3个）
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.23), ncol=4, fontsize=16, frameon=False)

# 调整边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 调整布局以防止标签重叠，为外部图例留出空间
plt.tight_layout()
plt.subplots_adjust(top=0.85)  # 为顶部图例留出空间

# 显示图表
plt.savefig(f'E:/cppwork/dco_benchmarks/DATA/figure/ALL/HNSW/hnsw_volume.png', dpi=400, bbox_inches='tight')    
plt.show()