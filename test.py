# -*- coding: utf-8 -*-
"""
@Author: Cinja
@Software: PyCharm
@File: test.py
@Date: 2022/6/3
"""
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

# # 导入数据
with open("B题附件/城市航线_时间.txt", "r", encoding='utf-8') as f:
    data = f.read()
    list = data.split()
    data_list = []
    for one in list:
        try:
            trans = int(one)
            data_list.append(trans)
        except:
            try:
                trans = float(one)
                data_list.append(trans)
            except:
                continue
    # print(data_list)

# 创建图对象
G = nx.Graph()
G2 = nx.Graph()

# 增加节点
G.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8])
G2.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8])

# 边颜色
edge_color = ['#6b5152' for i in range(15)]

# 增加权重，数据格式（节点1，节点2，权重）
for i in range(0, len(data_list), 3):
    G.add_edge(data_list[i], data_list[i+1], weight=data_list[i+2])

# for i in range(0, len(data_list), 3):
#     if data_list[i] == 8 or data_list[i+1] == 8:
#         G2.add_edge(data_list[i], data_list[i+1], weight=data_list[i+2])

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
# weights2 = nx.get_edge_attributes(G2, "weight")

# 画节点图
nx.draw_networkx(G, pos, with_labels=True, node_color='#dfd7d7', edge_color=edge_color)
# nx.draw_networkx(G2, pos, with_labels=True, node_color='#dfd7d7', edge_color='#0000FF', width=2.0)
# 画权重图
nx.draw_networkx_edge_labels(G, pos, edge_labels=weights)
# nx.draw_networkx_edge_labels(G2, pos, edge_labels=weights)

# 展示
plt.show()


# data = pd.read_excel('Results/Q2_Floyd_Distances.xls', index_col=0)
# print(data.describe())
# df = pd.DataFrame(data.describe())
# df.to_excel('Results/Q2_Floyd_Distances_Describe.xls')
# print(data)

