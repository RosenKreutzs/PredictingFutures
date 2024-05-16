# 导入必要的库
from keras.src import regularizers
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import numpy as np
import keras



def drop_split_future(data,drop_future,target):#删除无关特征，提取标签
    target_future=data[target]
    # 删除指定的列特征
    data = data.drop(drop_future, axis=1)
    data = data.round(2)
    data = data.drop(data.index[0])
    target_future = target_future.drop(X.index[0])
    return data,target_future

if __name__ == "__main__":
    df=pd.read_csv("../data_files/ADS/futures_data3.csv")
    X,y=drop_split_future(df,["commodity","date"],"tomorrow_close")
    X = X.drop(X.index[0])
    y = y.drop(X.index[0])
    # 将数据集分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train.shape)
    # 构建网络
    model = keras.models.Sequential()
    #regularizers.l2(0.01)表示L2正则化的强度，即权重参数的平方和会乘以0.01后加到损失函数中
    model.add(keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],),kernel_regularizer=regularizers.l2(0.01)))
    model.add(keras.layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
    model.add(keras.layers.Dense(1))  # 标量回归的典型操作(只是预测一个单一连续值的回归的最后一层)
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])  # mse是均方误差;mae是平均绝对误差(预测与真实值的差距)

    # 训练深度森林模型
    model.fit(X_train, y_train)
    # 使用训练好的模型进行预测
    print(X_train.shape)
    m=pd.DataFrame(X_train.iloc[0].values.reshape(1, 18))
    y_pred = model.predict(m)
    print(y_pred)
    with open('dnn_deepforest_model.pkl', 'wb') as file:
        pickle.dump(model, file)
