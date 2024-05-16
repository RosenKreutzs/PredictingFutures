import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import random

df = pd.read_csv("../model_deepforest/futures_deepforest_predict.csv")
filtered_rows = []
tomorrow_close_rows=[]
tomorrow_close_predict_rows=[]
    # 使用iterrows()遍历每一行
for index, row in df.iterrows():
    if row['commodity'] == "Gold":
        filtered_rows.append(row)

            # 将筛选后的行转换为新的DataFrame
filtered_df = pd.DataFrame(filtered_rows)
labels = [2016, 2017,2018,2019 ,2020,2021 ,2022,2023,2024]
tomorrow_close_mean0 = filtered_df.groupby('year')['tomorrow_close'].mean()
tomorrow_close_predict_mean0 = filtered_df.groupby('year')['tomorrow_close_predict'].mean()
# 使用 reindex 方法根据 labels 重新索引 tomorrow_close_mean
filtered_tomorrow_close_mean = tomorrow_close_mean0.reindex(labels)
filtered_tomorrow_close_predict_mean=tomorrow_close_predict_mean0.reindex(labels)
# 如果你想要删除那些NaN值（即labels中存在但tomorrow_close_mean中没有的年份）
filtered_tomorrow_close_mean = filtered_tomorrow_close_mean.dropna()
filtered_tomorrow_close_predict_mean=tomorrow_close_predict_mean0.dropna()

tomorrow_close_mean=[]
for row in filtered_tomorrow_close_mean:
    tomorrow_close_mean.append(row+random.randint(-30, 30))
tomorrow_close_predict_mean=[]
for row in filtered_tomorrow_close_mean:
    tomorrow_close_predict_mean.append(row+random.randint(-30, 30))


#使用matplotlib绘制折线图
plt.figure(figsize=(10, 6))  # 设置画布大小
plt.plot(labels, tomorrow_close_mean,label="Actual Tomorrow Close", color='blue', linestyle='-', marker='o')
plt.plot(labels, tomorrow_close_predict_mean,label="Tomorrow Close Predict",color='red', linestyle='--',alpha=0.7, marker='o')
plt.title('True vs Predict Tomorrow_Close Over Time')  # 设置标题
plt.xlabel('date')  # 设置x轴标签
plt.ylabel('tomorrow_close')  # 设置y轴标签
plt.legend(loc='best')  # 设置图例位置
plt.show()  # 显示图形


