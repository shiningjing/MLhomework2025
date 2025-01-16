import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, LSTM
from tcn import TCN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 加载数据
train_data = pd.read_csv('train_data.csv')  # 训练数据
test_data = pd.read_csv('test_data.csv')  # 测试数据
# 选择需要的列
train_values = train_data[["season","yr","mnth","hr","holiday","weekday","workingday",
                           "weathersit","temp","atemp","hum","windspeed","casual",
                           "registered",'cnt']][1:15200].values
test_values = test_data['cnt'][1:].values
# 数据标准化（归一化）
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_values = scaler.fit_transform(train_values.reshape(-1, 1))
scaled_test_values = scaler.transform(test_values.reshape(-1, 1))  # 使用训练集的缩放器来缩放测试集
# 创建滑动窗口数据，输入为过去240小时，输出为未来240小时
def create_dataset(data, time_step=240, forecast_horizon=240):
    X, y = [], []
    for i in range(len(data) - time_step - forecast_horizon + 1):
        X.append(data[i:(i + time_step), 0])  # 过去240小时的数据
        y.append(data[(i + time_step):(i + time_step + forecast_horizon), 0])  # 未来240小时的数据
    return np.array(X), np.array(y)
# 创建训练数据
X_train, y_train = create_dataset(scaled_train_values, time_step=240, forecast_horizon=240)
X_test, y_test = create_dataset(scaled_test_values, time_step=240, forecast_horizon=240)
# 重塑输入数据为LSTM所需的三维数组：[samples, time_steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
# 进行多次实验
results = []
n_experiments = 5
for i in range(n_experiments):
    print(f"第{i+1}轮实验")
    # 输入层
    input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))  # 输入为 (time_steps, features)
    # 卷积层
    conv_layer = Conv1D(filters=64, kernel_size=7, activation='relu')(input_layer)
    # TCN 编码器
    tcn_output = TCN(nb_filters=64,  # 卷积核数量
                     kernel_size=3,  # 卷积核大小
                     dilations=[1, 2, 4, 8],  # 扩张率
                     padding='causal',  # 因果卷积
                     dropout_rate=0.1,  # Dropout 概率
                     return_sequences=True)(conv_layer)
    # LSTM 层（如果需要）
    lstm_layer_1 = LSTM(64, return_sequences=True)(tcn_output)
    # 第二个 LSTM 层
    lstm_layer_2 = LSTM(64, return_sequences=False)(lstm_layer_1)
    # 输出层（调整维度）
    output_layer = Dense(240)(lstm_layer_2)  # 输出240个时间步，保持与目标形状一致
    # 构建模型
    model = Model(inputs=input_layer, outputs=output_layer)# 训练模型
    # 编译模型
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=50, batch_size=64)
    # 预测
    predicted = model.predict(X_test)
    # 逆标准化
    predicted = scaler.inverse_transform(predicted)
    y_test_actual = scaler.inverse_transform(y_test)
    # 计算评价指标
    mae = mean_absolute_error(y_test_actual, predicted)
    mse = mean_squared_error(y_test_actual, predicted)
    r2 = r2_score(y_test_actual, predicted)
    # 保存每次实验的结果
    results.append({'MAE': mae, 'MSE': mse, 'R2': r2})
    # 保存对比图
    plt.figure(figsize=(12, 6))
    # 将输入数据和预测数据连接起来
    # 输入数据是 X_test 的第一个样本，即过去240小时的数据
    # 预测数据是 predicted[0]，即预测的未来240小时的数据
    x_input = np.arange(0, 240)  # 输入数据的x轴
    x_pred = np.arange(240, 480)  # 预测数据的x轴

    plt.plot(x_input, scaler.inverse_transform(X_test[0].reshape(-1, 1)), label='输入数据 (历史数据)', color='green',
             linestyle='--')
    plt.plot(x_pred, predicted[0], label='预测值', color='red')
    plt.plot(x_pred, y_test_actual[0], label='真实值', color='blue')

    plt.title(f'第{i+1}轮实验：实际值与预测值对比')
    plt.xlabel('时间步')
    plt.ylabel('值')
    plt.legend()
    plt.savefig(f'comparison_{i+1}.png')
    plt.close()

# 转换为DataFrame
results_df = pd.DataFrame(results)

# 计算标准差
mae_std = results_df['MAE'].std()
rmse_std = results_df['MSE'].std()
r2_std = results_df['R2'].std()

# 将标准差添加到结果中
results_df['MAE_std'] = mae_std
results_df['MSE_std'] = rmse_std
results_df['R2_std'] = r2_std

# 保存结果到CSV文件
results_df.to_csv('experiment_results_with_std.csv', index=False)

# 打印标准差
print(f"MAE Standard Deviation: {mae_std}")
print(f"MSE Standard Deviation: {rmse_std}")
print(f"R-squared Standard Deviation: {r2_std}")

# 打印每轮的结果
print("\nExperiment Results:")
print(results_df)
