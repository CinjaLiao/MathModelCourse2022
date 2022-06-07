# -*- coding: utf-8 -*-
"""
@Author: Cinja
@Software: PyCharm
@File: Q1_drawGraph.py
@Date: 2022/5/13
"""

import networkx as nx
import matplotlib.pyplot as plt

# 导入数据
with open("B题附件/城市航线.txt", "r", encoding='utf-8') as f:
    data = f.read()
    list = data.split()
    data_list = []
    for one in list:
        try:
            trans = int(one)
            data_list.append(trans)
        except:
            continue
    # print(data_list)

# 创建图对象
G = nx.Graph()

# 增加节点
G.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8])

# 边颜色
edge_color = ['#6b5152' for i in range(15)]

# 增加权重，数据格式（节点1，节点2，权重）
for i in range(0, len(data_list), 3):
    G.add_edge(data_list[i], data_list[i+1], weight=data_list[i+2])

# 固定节点位置
pos = {
     1:[11, 11],
     2:[6, 11],
     3:[6, 0.5],
     4:[1, 8],
     5:[1, 3],
     6:[11, 1.5],
     7:[13, 7],
     8:[6, 5]
}

# 重新获取权重序列
weights = nx.get_edge_attributes(G, "weight")

# 画节点图
nx.draw_networkx(G, pos, with_labels=True, node_color='#dfd7d7', edge_color=edge_color)
# 画权重图
nx.draw_networkx_edge_labels(G, pos, edge_labels=weights)

# 展示
plt.show()
