# -*- coding: utf-8 -*-
"""
@Author: Cinja
@Software: PyCharm
@File: Q4_2_Path2Time.py
@Date: 2022/6/1
"""
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.tree import ExtraTreeRegressor
from dateutil import parser
from sklearn.preprocessing import StandardScaler

# 导入数据
def loadData(data_path):
    with open(data_path, "r", encoding='utf-8') as f:
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
    return data_list

# 画航线图
def drawFigure(data_list, new_path):
    # 画原始航线图(G1)、新增航线图(G2)
    # 转化并记录待标记路线
    path_num = len(new_path)
    # 创建图对象
    G1 = nx.Graph()
    G2 = nx.Graph()
    # 增加节点
    G1.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8])
    G2.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8])
    # 增加权重，数据格式（节点1，节点2，权重）
    for i in range(0, len(data_list), 3):
        G1.add_edge(data_list[i], data_list[i + 1], weight=data_list[i + 2])
    for i in range(path_num):
        G2.add_edge(new_path[i][0], new_path[i][1], weight=new_path[i][2])

    # 固定节点位置
    pos = {
        1: [12, 11],
        2: [6, 11],
        3: [8, 0.5],
        4: [1, 7],
        5: [1, 3],
        6: [11, 1.5],
        7: [13, 7],
        8: [6, 5]
    }
    # 重新获取权重序列
    weightsG1 = nx.get_edge_attributes(G1, "weight")
    weightsG2 = nx.get_edge_attributes(G2, "weight")
    # 画节点图、路径
    nx.draw_networkx(G1, pos, with_labels=True, node_color='#dfd7d7', edge_color='#6b5152', width=2.0)
    nx.draw_networkx(G2, pos, with_labels=True, node_color='#dfd7d7', edge_color='#0000FF')
    # 画权重图
    nx.draw_networkx_edge_labels(G1, pos, edge_labels=weightsG1, font_size=8)
    nx.draw_networkx_edge_labels(G2, pos, edge_labels=weightsG2, font_size=8)
    # 展示
    plt.show()

# 画单一航线图
def drawOneFigure(data_list):
    # 画航线图(G1)
    # 创建图对象
    G1 = nx.Graph()
    # 增加节点
    G1.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8])
    # 增加权重，数据格式（节点1，节点2，权重）
    for i in range(0, len(data_list), 3):
        G1.add_edge(data_list[i], data_list[i + 1], weight=data_list[i + 2])

    # 固定节点位置
    pos = {
        1: [11, 11],
        2: [6, 11],
        3: [6, 0.5],
        4: [1, 8],
        5: [1, 3],
        6: [11, 1.5],
        7: [13, 7],
        8: [6, 5]
    }
    # 重新获取权重序列
    weightsG1 = nx.get_edge_attributes(G1, "weight")
    # 画节点图
    nx.draw_networkx(G1, pos, with_labels=True, node_color='#dfd7d7', edge_color='#6b5152')
    # 画权重图
    nx.draw_networkx_edge_labels(G1, pos, edge_labels=weightsG1)
    # 展示
    plt.show()

# 导出数据
def writeData(data_path_out, data_list):
    # 新建列表
    write_data = []
    write_headers = ['%数据集：八个城市之间的航线数据', '%作者：cinja', '%格式：起点，终点，时间/h']

    # 处理data_list
    for i in range(0, len(data_list), 3):
        one_str = str(data_list[i]) + ' ' + str(data_list[i+1]) + ' ' + str(data_list[i+2])
        write_data.append(one_str)

    # 合并列表
    write_data = write_headers + write_data

    # 写数据
    with open(data_path_out, "w", encoding='utf-8') as f:
        for i in write_data:
            f.writelines(i)
            f.write('\n')


def loadData_ML(path):
    # 加载数据
    df = pd.read_excel(path, usecols=['mileage', 'departure_time', 'landing_time'])
    df = df.drop(df[df.mileage == 0].index)
    df = df.drop(df[df.mileage > 1500].index)
    df = df.reset_index(drop=True)
    df['time'] = 0
    for i in range(df.shape[0]):
        df.loc[i, 'time'] = (parser.parse(str(df.loc[i, 'landing_time'])) - parser.parse(str(df.loc[i, 'departure_time']))).seconds

    return df

def generateTrain(df):
    # 生成训练集和测试集
    # df = washData(df)
    data = df.loc[:, 'mileage']
    label = df.loc[:, 'time']
    # 数据切分
    X_train = data
    y_train = label

    return X_train, y_train

def transform2Hour(time_list):
    time_list = time_list / 3600
    for i in range(len(time_list)):
        time_list[i] = round(time_list[i], 2)
    return time_list



if __name__ == "__main__":
    # 初始化参数
    distance_path = 'Results/Distances.xls'
    data_path = "B题附件/城市航线.txt"
    data_path_out = "B题附件/城市航线_时间.txt"
    CITY_NUM = 8

    # 导入数据
    distance_mat = pd.read_excel(distance_path, index_col=0)
    data_list = loadData(data_path)
    distance_list = [data_list[x] for x in range(2, len(data_list), 3)]

    # 导入机器学习数据
    path = 'Data\国内航班数据.xls'
    data = loadData_ML(path=path)
    X_train, y_train = generateTrain(data)

    # 归一化
    standardScaler = StandardScaler()
    X_train_fit = standardScaler.fit_transform(X_train.values.reshape(-1, 1))
    X_test_fit = standardScaler.transform(np.array(distance_list).reshape(-1, 1))

    # 生成机器学习对象
    model_ExtraTreeRegressor = ExtraTreeRegressor()
    # 训练
    model_ExtraTreeRegressor.fit(X_train_fit, y_train)
    # 回归
    y_pred = model_ExtraTreeRegressor.predict(X_test_fit)

    # 把时间转换为小时
    time_list = transform2Hour(y_pred)

    # 把data_list中的距离替换为时间
    j = 0
    for i in range(2, len(data_list), 3):
        data_list[i] = time_list[j]
        j += 1

    # 画航线图
    drawOneFigure(data_list)

    # 导出航线数据
    # writeData(data_path_out, data_list)

