import matplotlib.pyplot as plt
import numpy as np

# 数据集（X轴）
datasets = ['Contriever', 'DEEP', 'GIST', 'GloVe', 'Instructorxl', 'MSong', 'OpenAI', 'SIFT']
n_datasets = len(datasets)

# 方法（行名）
methods_query_process = [
    'ADS', 'DADE', r"DDC$_{res}$", r"DDC$_{pca}$", r"DDC$_{opq}$", 'RaBitQ','Finger','Flash'
]

# 表格数据（单位 us）
data_query_process = {
    'ADS': [218.39, 4.11, 343.34, 15.86, 218.75, 65.77, 879.05, 6.90],
    'DADE': [112.45, 2.47, 175.55, 9.06, 113.99, 35.58, 456.78, 4.13],
    r"DDC$_{res}$": [212.5, 4.11, 343.34, 15.86, 218.75, 65.77, 879.05, 6.90],
    r"DDC$_{pca}$": [214.16, 4.11, 343.34, 15.86, 218.75, 65.77, 879.05, 6.90],
    r"DDC$_{opq}$": [215.17, 4.47, 331.12, 27.09, 214.07, 71.26, 843.35, 7.29],
    'RaBitQ': [15, 4, 15, 6, 9, 8, 29, 6],
    'Finger': [19.75, 2.77, 24.7, 5.3, 19.73, 10.80, 40.12, 3.43],
    'Flash': [6,6,18,4,6,9,21,6],
}

# 样式（颜色可自行调整）
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


# 绘图函数
def draw_chart(ax, data, methods, title, ylabel):
    n_methods = len(methods)
    bar_width = 0.8 / n_methods
    index = np.arange(n_datasets)
    positions_list = []

    # 画柱子
    for i, method in enumerate(methods):
        positions = index + (i - n_methods / 2 + 0.5) * bar_width
        ax.bar(
            positions, data[method], width=bar_width,
            label=method, color=styles[method]['color'],
            edgecolor='black', hatch=styles[method]['hatch'], linewidth=1
        )
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
        ax.plot(bar_x, bar_height * 1.2, 'r*', markersize=7, markeredgecolor='red', 
                markerfacecolor='red', markeredgewidth=0.5)

    # y轴对数刻度
    ax.set_yscale('log')
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_xticks(index)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.tick_params(axis='both', which='major', labelsize=11)

    # 图例放上方
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=4, frameon=False, fontsize=13, columnspacing=2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# 创建画布
fig, ax = plt.subplots(1, 1, figsize=(8, 7))
draw_chart(ax, data_query_process, methods_query_process, '(3) Query & Processing Time', 'Time Consumption (us)')

plt.tight_layout(rect=[0, 0.2, 1, 0.85])
plt.savefig(f'E:/cppwork/dco_benchmarks/DATA/figure/ALL/HNSW/query_preprocessing_time.png', dpi=400, bbox_inches='tight',format='png')
plt.show()
