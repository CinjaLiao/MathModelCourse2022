# -*- coding: utf-8 -*-
"""
@Author: Cinja
@Software: PyCharm
@File: Q4_2_MachineLearningModels.py
@Date: 2022/6/1
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dateutil import parser
import re


def loadData(path):
    # 加载数据
    df = pd.read_excel(path, usecols=['mileage', 'departure_time', 'landing_time'])
    df = df.drop(df[df.mileage == 0].index)
    # df = df.drop(df[df.mileage > 1500].index)
    df = df.reset_index(drop=True)
    df['time'] = 0
    for i in range(df.shape[0]):
        df.loc[i, 'time'] = (parser.parse(str(df.loc[i, 'landing_time'])) - parser.parse(str(df.loc[i, 'departure_time']))).seconds

    return df


def generateTrainAndTest(df):
    # 生成训练集和测试集
    # df = washData(df)
    data = df.loc[:, 'mileage']
    label = df.loc[:, 'time']
    # 数据切分
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def model_function(regr):
    model_name = re.findall(r'(.*?)[(].*?', str(regr))[0]

    regr.fit(X_train_fit, y_train)
    y_pred = regr.predict(X_test_fit)

    RMSE = metrics.mean_squared_error(y_test, y_pred, sample_weight=None, multioutput='uniform_average') ** 0.5
    MAE = metrics.mean_absolute_error(y_test, y_pred)
    r2_score = metrics.r2_score(y_test, y_pred)

    # one_result = {}
    one_result = dict([['RMSE', RMSE], ['MAE', MAE], ['r2', r2_score]])
    # for i in [RMSE, MAE, r2_score]:
    #     one_result.update({str(i): i})
    result.update({str(model_name): one_result})

    print('='*20)
    print('回归模型：', str(model_name))
    print('RMSE(均方根误差)：', RMSE)
    print('MAE(平均绝对误差)：', MAE)
    print('r2(决定系数)：', r2_score)
    print('='*20)


path = 'Data\国内航班数据.xls'
result = {}
data = loadData(path=path)
X_train, X_test, y_train, y_test = generateTrainAndTest(data)

# 归一化
standardScaler = StandardScaler()
X_train_fit = standardScaler.fit_transform(X_train.values.reshape(-1,1))
X_test_fit = standardScaler.transform(X_test.values.reshape(-1,1))


# 回归模型来源：
# https://blog.csdn.net/ChenVast/article/details/82107490
# https://cloud.tencent.com/developer/article/1419988

# Linear Regression
from sklearn import linear_model, metrics
model_LinearRegression = linear_model.LinearRegression()

# Decision Tree Regressor
from sklearn import tree
model_DecisionTreeRegressor = tree.DecisionTreeRegressor()

# SVM Regressor
from sklearn import svm
model_SVR = svm.SVR()

# K Neighbors Regressor
from sklearn import neighbors
model_KNeighborsRegressor = neighbors.KNeighborsRegressor()

# Random Forest Regressor
from sklearn import ensemble
model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20)

# Adaboost Regressor
from sklearn import ensemble
model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators=50)

# Gradient Boosting Random Forest Regressor
from sklearn import ensemble
model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=100)

# bagging Regressor
from sklearn.ensemble import BaggingRegressor
model_BaggingRegressor = BaggingRegressor()

# ExtraTree Regressor
from sklearn.tree import ExtraTreeRegressor
model_ExtraTreeRegressor = ExtraTreeRegressor()

# 岭回归
from sklearn.linear_model import Ridge
model_Ridge = Ridge(alpha=.5)

# Lasso回归
from sklearn.linear_model import Lasso
model_Lasso = Lasso(alpha=0.1)

# Elastic Net 回归
from sklearn.linear_model import ElasticNet
model_ElasticNet = ElasticNet(random_state=0)

# 贝叶斯岭回归
from sklearn.linear_model import BayesianRidge
model_BayesianRidge = BayesianRidge()

# SGD 回归
from sklearn.linear_model import SGDClassifier
model_SGDClassifier = SGDClassifier(max_iter=1000, tol=1e-3)

# 神经网络
from sklearn.neural_network import MLPRegressor
model_MLPRegressor = MLPRegressor()

# XGBoost 回归
import xgboost as xgb
model_XGBRegressor = xgb.XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100, objective='reg:linear', n_jobs=-1)

# LightGBM 回归
import lightgbm as lgb
model_LGBMRegressor = lgb.LGBMRegressor(num_leaves=31, learning_rate=0.05, n_estimators=20)


model_function(model_LinearRegression)
model_function(model_DecisionTreeRegressor)
model_function(model_SVR)
model_function(model_KNeighborsRegressor)
model_function(model_RandomForestRegressor)
model_function(model_AdaBoostRegressor)
model_function(model_GradientBoostingRegressor)
model_function(model_BaggingRegressor)
model_function(model_ExtraTreeRegressor)
model_function(model_Ridge)
model_function(model_Lasso)
model_function(model_ElasticNet)
model_function(model_BayesianRidge)
model_function(model_SGDClassifier)
model_function(model_MLPRegressor)
model_function(model_XGBRegressor)
model_function(model_LGBMRegressor)

# print(result)
# result_df = pd.DataFrame(result)
# result_df_T = pd.DataFrame(result_df.values.T,index=result_df.columns,columns=result_df.index)
# result_df_T.to_excel('Results/MachineLearningCompare.xls')
