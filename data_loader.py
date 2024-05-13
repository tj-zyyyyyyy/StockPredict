import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from utils.timefeatures import time_features
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class StandardScaler(object):
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std


class Dataset_Custom(Dataset):
    def __init__(self, root_path, tweet_path, time_length, flag='train', size=None):
        # size [seq_len,pred_len]
        self.seq_len = size[0]  # 使用的数据长度
        self.pred_len = size[1]  # 预测的长度
        self.time_length = time_length

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.root_path = root_path
        self.tweet_path = tweet_path
        self.__read_data__()  # 调用__read_data__函数读取数据

    # 一次读一个股票
    def __read_data__(self):
        self.scaler = StandardScaler()
        stock_df_raw = {}
        filenames = os.listdir(self.root_path)
        length = self.time_length
        train_len = int(length * 0.7)
        test_len = int(length * 0.2)
        val_len = int(length * 0.1)

        start_borders = [0, train_len - self.seq_len, train_len + val_len - self.seq_len]  # 只要预测部分不重叠
        end_borders = [train_len, train_len + val_len, train_len + val_len + test_len]
        start_border = start_borders[self.set_type]
        end_border = end_borders[self.set_type]
        non_date_data = []
        date_data = []
        df_stamp = []

        for file in filenames:
            df_raw = pd.read_csv(os.path.join(self.root_path, file))
            stock_df_raw[file.split('.')[0]] = df_raw
            non_date_data.append(df_raw[df_raw.columns[1:-1]].values)  # 使用.values将DataFrame转换为NumPy数组
            df_stamp.append(pd.DatetimeIndex(df_raw[start_border:end_border]['date']).values)
            date_data.append(df_raw[start_border:end_border]['date'].values)

        # 将列表转换为NumPy数组
        non_date_data = np.array(non_date_data)
        df_stamp = np.array(df_stamp)
        date_data = np.array(date_data)

        # train_data = non_date_data[:, start_borders[0]:end_borders[0],:]
        # 用训练集的mean和std来标准化数据集
        # self.scaler.fit(train_data)
        # data = self.scaler.transform(non_date_data)
        # 提取日期特征
        data_stamp = time_features(df_stamp)
        # 在下面分seq_len和pred_len
        self.data = non_date_data[:, start_border:end_border, :]
        self.data_stamp = data_stamp

        # 获取对应新闻
        tokenizer = AutoTokenizer.from_pretrained("model/finbert")
        tweet_data = []
        for file in filenames:
            onestock_tweet_data = []
            tweet_files_path = os.path.join(self.tweet_path, file.split('.')[0])
            for date in date_data[0]:
                tweet_set = set()
                tweet_data_oneday = str()
                tweet_file_path = os.path.join(tweet_files_path, date)
                if os.path.exists(tweet_file_path):  # 该日期的推特是否存在
                    with open(tweet_file_path, 'r', encoding='utf-8') as file:
                        for line in file:
                            # 去除每行末尾的换行符，并尝试解析为JSON对象
                            json_data = json.loads(line.strip())
                            if json_data['text'] not in tweet_set:
                                tweet_data_oneday += json_data['text']
                            tweet_set.add(json_data['text'])
                onestock_tweet_data.append(tweet_data_oneday)
            tweet_data.append(onestock_tweet_data)

        tweet_data_padded = [tokenizer(sublist, return_tensors="pt", padding=True, truncation=True).data for sublist in tweet_data]  # token
        self.tweet_data = tweet_data_padded

    def __getitem__(self, index):
        seq_begin = index
        seq_end = seq_begin + self.seq_len  # 数据预测部分
        pred_begin = seq_end
        pred_end = pred_begin + self.pred_len  # 预测部分

        data_x = self.data[:, seq_begin:seq_end, :]
        data_y = self.data[:, pred_begin:pred_end, -1:]
        data_x_stamp = self.data_stamp[:, seq_begin:seq_end, :]
        data_y_stamp = self.data_stamp[:, pred_begin:pred_end, :]

        tweet_data_x = []
        for one_stock_tweet in self.tweet_data: # dict [time,tweet]
            one_stock_dict = {}
            for key,val in one_stock_tweet.items():
                one_stock_dict[key] = val[seq_end-1] # 新闻只取前一天的
            tweet_data_x.append(one_stock_dict)

        return data_x, data_y, data_x_stamp, data_y_stamp, tweet_data_x
        # [seq_len,pred_len,seq_len,pred_len]

    # 可以从数据中提取的有效序列的数量
    def __len__(self):
        return len(self.data[0]) - self.seq_len - self.pred_len + 1
