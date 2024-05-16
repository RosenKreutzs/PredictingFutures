# -*- coding: utf-8 -*-
import numpy as np
from deepforest import CascadeForestRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV

def drop_split_future(data,drop_future,target):#删除无关特征，提取标签
    target_future=data[target]
    # 删除指定的列特征
    data = data.drop(drop_future, axis=1)
    data = data.round(2)
    return data,target_future

if __name__ == "__main__":
    df=pd.read_csv("../data_files/ADS/futures_data3.csv")
    X,y=drop_split_future(df,["commodity","date"],"tomorrow_close")
    # 将数据集分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 创建深度森林对象
    forest = CascadeForestRegressor()

    """
        使用了RandomizedSearchCV函数来执行随机搜索，
        并指定了参数空间param_dist。n_iter参数指定了随机搜索中要尝试的参数组合数量，
        cv参数指定了交叉验证的折数，
        scoring参数则指定了评估指标。执行完随机搜索后，
        我们可以从best_params_属性中获取最佳参数组合，
        并使用best_estimator_属性获取使用这些参数训练的模型。
    """
    param_dist = {
        "n_estimators": randint(low=10, high=100),#树的数量
        "max_depth": randint(low=1, high=3),#每个树的最大深度
        "max_layers":randint(low=5, high=20),#级联森林的最大层数
        "n_trees": randint(low=5, high=100),#在级联森林的每一层中，每个随机森林包含的树的数量
        "n_tolerant_rounds": randint(low=1, high=10),#容忍连续多少个回合的层生长性能不提升，然后停止增加更多层
    }
    # 创建Random Search对象
    random_search = RandomizedSearchCV( forest,#deepforest模型对象
                                        param_distributions=param_dist,#超参数的取值空间
                                        n_iter=100,#随机搜索中要尝试的参数组合数量
                                        cv=5,#交叉验证的折数
                                        scoring='accuracy')#scoring参数则指定了评估指标

    # 输出最佳参数组合
    random_search.fit(X, y)
    print("Best parameters set found on development set:")
    print()
    print(random_search.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = random_search.cv_results_['mean_test_score']
    stds = random_search.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, random_search.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    # 使用最佳参数重新训练模型

    best_clf = random_search.best_estimator_

    best_clf.fit(X_train, y_train)

    # 使用训练好的模型进行预测
    y_pred = best_clf.predict(X_test)
    # 保存模型
    best_clf.save()