# 导入必要的库
from keras import Input
from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.models import Model
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import numpy as np
from keras.layers import Concatenate


def drop_split_future(data,drop_future,target):#删除无关特征，提取标签
    target_future=data[target]
    # 删除指定的列特征
    data = data.drop(drop_future, axis=1)
    data = data.round(2)
    data = data.drop(1)
    target_future = target_future.drop(target_future.index[0])
    return data,target_future

if __name__ == "__main__":
    df=pd.read_csv("data_files/ADS/futures_data3.csv")
    X,y=drop_split_future(df,["result"],"result")
    print(X)
    print(y)