import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression

# 获取文件夹中的所有csv文件
folder_path = './data/price'
files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

count_0 = 0
count_1 = 0
length = 4

for file in files:
    print(file)
    # 读取csv文件
    df = pd.read_csv(os.path.join(folder_path, file))

    # 初始化新列
    df['label_for_week'] = -1

    # 对每一行进行操作
    for i in range(len(df) - length + 1):
        # 获取第i至第i+6行的closing_price_for_label
        y = df.loc[i:i + length -1, 'closing_price_for_label'].values.reshape(-1, 1)
        X = np.array(range(length)).reshape(-1, 1)

        # 线性拟合
        reg = LinearRegression().fit(X, y)

        # 如果斜率k>=0，将label_for_week设为1
        if reg.coef_[0] >= 0:
            df.loc[i, 'label_for_week'] = 0
            count_1 += 1
        else:
            count_0 += 1

    # 删除最后六行
    df = df.iloc[:-length+1]

    # 保存修改后的csv文件
    df.to_csv(os.path.join(folder_path, file), index=False)

print(count_0)
print(count_1)
