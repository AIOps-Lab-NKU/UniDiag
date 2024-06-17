import numpy
import pandas as pd

# hyperparameter: path, win_size= 3600, std_thr = 0.1, k_thr = 3
path = './metric/label_ori_metric/node/'

def k_sigma(path, win_size, std_thr, k_thr):
    df = pd.read_csv(path)
    label = []
    for i in range(len(df)):
        window = df.loc[(df['timestamp']>=df['timestamp'][i]-win_size)&(df['timestamp']<df['timestamp'][i])]
        if len(window) == 0:
            label.append(3)
        else:
            value_list = window['value']
            value_threshold = (numpy.nanmean(value_list),max(numpy.nanstd(value_list), std_thr*numpy.nanmean(value_list)))
            if value_threshold[0] + k_thr * value_threshold[1] < df['value'][i]:
                label.append(1)
            elif  value_threshold[0] - k_thr * value_threshold[1] > df['value'][i]:
                label.append(2)
            else:
                label.append(0)
    df['label'] = label
    timestamp_list = []
    value_list = []
    label_list = []
    for i in range(len(df)):
        if df['label'][i] != 3:
            timestamp_list.append(df['timestamp'][i])
            value_list.append(df['value'][i])
            label_list.append(df['label'][i])
    res_df = pd.DataFrame({'timestamp': timestamp_list, 'value': value_list, 'label': label_list})
    return res_df