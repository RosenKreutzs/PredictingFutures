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
    return data,target_future

if __name__ == "__main__":
    df=pd.read_csv("../data_files/ADS/futures_data3.csv")
    X1,y1=drop_split_future(df,["commodity","date","tomorrow_close"],"tomorrow_close")
    X1 = X1.drop(X1.index[0])
    X0=X1.to_numpy()
    y1 = y1.drop(X1.index[0])
    y0=y1.to_numpy()
    # 将数据集分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
    df1 = df.drop(["commodity","date"], axis=1)
    df1 = df1.round(2)
    df_numpy=df1.to_numpy()

    """
从原理和应用场景上来看，GAN模型并不直接适用于回归问题。
但我将生成器generator的生成对象设为预测特征tomorrow_close，
再将判别器discriminator的输入对象设为生成器的生成对象与其余18个特征的拼接向量，
就能实现GAN模型对于回归问题的预测了。
    """
    # 构建生成器
    generator_input = Input(shape=(17,))
    x = Dense(128)(generator_input)
    x = LeakyReLU(alpha=0.2)(x)
    generator_output = Dense(1, activation='tanh')(x)
    generator = Model(generator_input, generator_output)

    # 编译生成器
    generator.compile(optimizer='rmsprop', loss='mse')

    # 构建判别器
    discriminator_input = Input(shape=(18,))  # 假设你想拼接后的输入是19维
    x = Dense(128)(discriminator_input)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.3)(x)
    discriminator_output = Dense(1, activation='sigmoid')(x)
    discriminator = Model(discriminator_input, discriminator_output)

    # 编译判别器
    discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 构建 GAN 模型
    discriminator.trainable = False
    generator.trainable = True
    gan_input = generator_input  # GAN模型的输入与生成器的输入相同
    gan_output = discriminator(Concatenate()([generator_output, generator_input]))
    gan = Model(gan_input, gan_output)

    # 编译GAN模型
    gan_optimizer = Adam()
    gan.compile(loss='binary_crossentropy', optimizer=gan_optimizer)

# 训练 GAN 模型
    epochs = 1000
    batch_size = 128
    sample_interval = 100

    # 初始化一些变量来保存训练过程中的损失
    d_loss_real = []
    d_loss_fake = []
    g_loss = []
    print(df_numpy)
    for epoch in range(epochs):
        # ---------------------
        #  训练判别器
        # ---------------------
        discriminator.trainable = True
        generator.trainable = False
        # 选择真实样本和噪声样本
        idx = np.random.randint(0, df_numpy.shape[0], batch_size)
        imgs_real = df_numpy[idx]  # 使用iloc基于整数位置索引行
        imgs_real.astype(np.float32)
        noise = np.random.normal(0, 1, (batch_size, 17))
        # 生成“假”样本
        imgs_fake0 = generator.predict(noise)
        imgs_fake = np.concatenate((noise, imgs_fake0), axis=1)
        # 训练判别器识别真实样本

        d_loss_real_curr = discriminator.train_on_batch(imgs_real, np.ones((batch_size, 1)))#inputing shape (None, 19)
        # 训练判别器识别“假”样本

        d_loss_fake_curr = discriminator.train_on_batch(imgs_fake, np.zeros((batch_size, 1)))
        # 计算判别器的总损失
        d_loss = 0.5 * np.add(d_loss_real_curr, d_loss_fake_curr)

        # ---------------------
        #  训练生成器
        # ---------------------
        discriminator.trainable = False
        generator.trainable = True
        # 生成器希望判别器认为其输出是真实的
        idx0 = np.random.randint(0, X0.shape[0], batch_size)
        X0_real = X0[idx0]  # 使用iloc基于整数位置索引行
        X0_real.astype(np.float32)
        y0_real=y0[idx0]
        y0_real.astype(np.float32)
        # 训练生成器
        g_loss_curr = generator.train_on_batch(X0_real, y0_real)
        # 记录损失
        d_loss_real.append(d_loss_real_curr)
        d_loss_fake.append(d_loss_fake_curr)
        g_loss.append(g_loss_curr)
        # 打印和绘制进度条
        print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss[0]))

    with open('futures_gan_model.pkl', 'wb') as file:
        pickle.dump(generator, file)



