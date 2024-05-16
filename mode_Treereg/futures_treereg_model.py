import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle


def drop_split_future(data, drop_features, target):
    target_future = data[target]
    data = data.drop(drop_features, axis=1)
    return data, target_future


if __name__ == "__main__":
    df = pd.read_csv("../data_files/ADS/futures_data3.csv")
    X, y = drop_split_future(df, ["commodity", "date"], "tomorrow_close")
    X = X.drop(X.index[0])  # 删除第一行数据
    y = y.drop(X.index[0])  # 删除对应的目标值

    # 将数据集分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 构建决策树回归模型
    tree_reg = DecisionTreeRegressor(random_state=42)  # 设置随机种子以便结果可复现

    # 训练模型
    tree_reg.fit(X_train, y_train)

    # 使用训练好的模型进行预测
    y_pred = tree_reg.predict(X_test)

    # 评估模型
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")

    # 保存模型
    with open('tree_reg_model.pkl', 'wb') as file:
        pickle.dump(tree_reg, file)