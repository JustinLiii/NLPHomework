# coding=utf-8
# 将数据集转为json格式

import json
import os
import pandas

train_fullname = 'data/train_data_public.csv'
test_fullname = 'data/test_public.csv'

train_data = pandas.read_csv(train_fullname)
test_data = pandas.read_csv(test_fullname)

# 将训练集转为json格式
train_data_json = []
for i in range(len(train_data)):
    data = dict()
    data['text'] = train_data['text'][i].strip()
    if train_data['class'][i] == 1:
        data['class'] = '正面'
    elif train_data['class'][i] == 0:
        data['class'] = '负面'
    elif train_data['class'][i] == 2:
        data['class'] = '中立'
    train_data_json.append(data)
# write with utf-8
with open('data/train_data_public.json', 'w', encoding='utf-8') as f:
    json.dump(train_data_json, f, ensure_ascii=False)

# 测试集
test_data_json = []
for i in range(len(test_data)):
    test_data_json.append({'text': train_data['text'][i].strip()})

with open('data/test_public.json', 'w', encoding='utf-8') as f:
    json.dump(test_data_json, f, ensure_ascii=False)
