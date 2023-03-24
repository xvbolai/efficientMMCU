import os
import argparse
import numpy as np
from sklearn.preprocessing import scale
from base.metric import *
from utils.log_utils import *
import pandas as pd
from lr_fit import lr_fit, lr_test, lr_model_to_json, save_model


def linear_regression(data, x_names, target, file, frac = 0.80, random_state = 100):
    train = data.sample(frac=frac, random_state=random_state, axis=0)
    test = data[~data.index.isin(train.index)]

    x = train[x_names].to_numpy()
    y = train[target].to_numpy()
    LogUtil.print_log('Features: {}'.format(x_names), LogLevel.DEBUG)
    LogUtil.print_log('Target: {}'.format(target),  LogLevel.DEBUG)
    # lr_train_type = 'grad_desc'
    lr_train_type = 'sklearn'
    do_x_scale = False
    do_y_scale = False
    show_fig = False

    acc, clf, means, stds = lr_fit(x, y, lr_type=lr_train_type, x_scale=do_x_scale, y_scale=do_y_scale,show_coef=False, show_fig=show_fig)

    print(acc)

    x = test[x_names].to_numpy()
    y = test[target].to_numpy()

    acc = lr_test(clf, x, y,
                    x_scale=do_x_scale, y_scale=do_y_scale,
                    means=means, stds=stds)


    json_text = lr_model_to_json(clf, lr_train_type, means, stds)

    # print('JSON text: ' + json_text + '\nacc:', acc)
    LogUtil.print_log('JSON text: ' + json_text, LogLevel.DEBUG)
    save_model('linear_regression_parameters', file + '.json', json_text)


def convolution_kernelN1_HWC(file):
    df = pd.read_csv(file)
    
    df['FLOPs'] = df['OH'] * df['OW'] * df['OC'] * (2 * df['KH'] * df['KW'] * df['IC'] + 1)
    # filter
    df = df[df['FLOPs'] > 3]
    df = df[df['T'] > 40]
    # per OP time
    # df['FLOPs_t'] = df['T'] / df['FLOPs']
    df['M'] = df['OH'] * df['OW']
    df['K'] = df['IC'] * df['KH'] * df['KW']
    df['N'] = df['OC']
    df['MN'] = df['M'] * df['N']
    df['KN'] = df['K'] * df['N']
    df['MK'] = df['M'] * df['K']
    df['MKN'] = df['M'] * df['K'] * df['N']
    
    # lable 
    x_names = ['M', 'K', 'N', 'MK', 'KN', 'MN', 'MKN']
    target = 'T'
    linear_regression(df, x_names, target, file)

def convolution_kernelN1_CHW(file):
    df = pd.read_csv(file)
    
    df['FLOPs'] = df['OH'] * df['OW'] * df['OC'] * (2 * df['KH'] * df['KW'] * df['IC'] + 1)
    # filter
    df = df[df['FLOPs'] > 3]
    df = df[df['T'] > 40]
    # per OP time
    # df['FLOPs_t'] = df['T'] / df['FLOPs']
    df['M'] = df['OH'] * df['OW']
    df['K'] = df['IC'] * df['KH'] * df['KW']
    df['N'] = df['OC']
    df['MN'] = df['M'] * df['N']
    df['KN'] = df['K'] * df['N']
    df['MK'] = df['M'] * df['K']
    df['MKN'] = df['M'] * df['K'] * df['N']
    df['KHKWOH'] = df['KH'] * df['KW'] * df['OH']
    df['KHKWOW'] = df['KH'] * df['KW'] * df['OW']
    df['KHKWOC'] = df['KH'] * df['KW'] * df['OC']
    # lable 
    x_names = ['KH', 'KW', 'KHKWOH', 'KHKWOW', 'KHKWOC', 'M', 'K', 'N', 'MK', 'KN', 'MN', 'MKN']
    target = 'T'
    linear_regression(df, x_names, target, file, frac=0.83, random_state=100)

def convolution_kernel1_stride1_pad0_HWC(file):
    df = pd.read_csv(file)
    # FLOPs
    df['FLOPs'] = df['OH'] * df['OW'] * df['OC'] * (2 * df['KH'] * df['KW'] * df['IC'] + 1)
    # filter
    df = df[df['FLOPs'] > 3]
    df = df[df['T'] > 40]
    # per OP time
    # df['FLOPs_t'] = df['T'] / df['FLOPs']
    df['M'] = df['OH'] * df['OW']
    df['K'] = df['IC'] * df['KH'] * df['KW']
    df['N'] = df['OC']
    df['MN'] = df['M'] * df['N']
    df['KN'] = df['K'] * df['N']
    df['MK'] = df['M'] * df['K']
    df['MKN'] = df['M'] * df['K'] * df['N']
    
    # lable 
    x_names = ['M', 'K', 'N', 'MK', 'KN', 'MN', 'MKN']
    target = 'T'
    linear_regression(df, x_names, target, file)
    
def convolution_kernel1_stride1_pad0_CHW(file):
    df = pd.read_csv(file)
    # FLOPs
    df['FLOPs'] = df['OH'] * df['OW'] * df['OC'] * (2 * df['KH'] * df['KW'] * df['IC'] + 1)
    # filter
    df = df[df['FLOPs'] > 3]
    df = df[df['T'] > 40]
    # per OP time
    # df['FLOPs_t'] = df['T'] / df['FLOPs']
    df['M'] = df['OH'] * df['OW']
    df['K'] = df['IC'] * df['KH'] * df['KW']
    df['N'] = df['OC']
    df['MN'] = df['M'] * df['N']
    df['KN'] = df['K'] * df['N']
    df['MK'] = df['M'] * df['K']
    df['MKN'] = df['M'] * df['K'] * df['N']
    
    # lable 
    x_names = ['M', 'K', 'N', 'MK', 'KN', 'MN', 'MKN']
    target = 'T'
    linear_regression(df, x_names, target, file)

def depthwise_HWC_or_CHW(file):
    df = pd.read_csv(file)
    # FLOPs
    df['FLOPs'] = (2 * df['KH'] * df['KW'] + 1) * df['OH'] * df['OW'] * df['OC']
    # filter
    df = df[df['FLOPs'] > 3]
    df = df[df['T'] > 40]
    # per OP time
    df['FLOPs_t'] = df['T'] / df['FLOPs']
    
    # lable 
    x_names = ['OH','OW', 'OC']
    target = 'FLOPs_t'
    linear_regression(df, x_names, target, file)

def avepool_q7_HWC(file):
    df = pd.read_csv(file)
    # df = df[df['KH'] == 2]
    # FLOPs
    df['FLOPs'] = (2 * df['KH'] * df['KW'] + 1) * df['OH'] * df['OW'] * df['OC']
    # filter
    df['IHP'] = df['IH'] + 2 * df['PH']
    df['IWP'] = df['IW'] + 2 * df['PW']
    df['PARAMS'] = df['IHP'] * df['IWP'] * df['IC']
    df['OHOW'] = df['OH'] * df['OW']
    df['OHOC'] = df['OH'] * df['OC']
    df['OWOC'] = df['OW'] * df['OC']
    df['OHOWOC'] = df['OH'] * df['OW'] * df['OC']
    df = df[df['FLOPs'] > 3]
    df = df[df['T'] > 20]
    # per OP time 统一一下
    df['FLOPs_t'] = df['T'] / df['FLOPs'] * 15 / 25
    
    # lable 
    x_names = ['OH', 'OW', 'OC', 'OHOW', 'OHOC', 'OWOC', 'OHOWOC']
    # x_names = ['IHP', 'IWP', 'IC', 'PARAMS']
    # x_names = ['OH', 'OW', 'OC', 'OHOWOC']
    target = 'T'
    linear_regression(df, x_names, target, file, frac = 0.80, random_state = 90)

def avgpooling_CHW_or_HWC(file):
    df = pd.read_csv(file)
    df['FLOPs'] = (2 * df['KH'] * df['KW'] + 1) * df['OH'] * df['OW'] * df['OC']
    df = df[df['FLOPs'] > 3]
    df = df[df['T'] > 2]
    df['KHKW'] = df['KH'] * df['KW']
    df['OHOW'] = df['OH'] * df['OW']
    df['OHOC'] = df['OH'] * df['OC']
    df['OWOC'] = df['OW'] * df['OC']
    df['OHOWOC'] = df['OH'] * df['OW'] * df['OC']
    df['KHKWOH'] = df['KHKW'] * df['OH']
    df['KHKWOW'] = df['KHKW'] * df['OW']
    df['KHKWOC'] = df['KHKW'] * df['OC']
    x_names = ['OH', 'OW', 'OC', 'KH', 'KW', 'KHKW', 'OHOW', 'OHOC', 'OWOC', 'OHOWOC', 'KHKWOH', 'KHKWOW', 'KHKWOC']
    target = 'T'
    linear_regression(df, x_names, target, file, frac = 0.60, random_state = 70)



def maxpooling_CHW_or_HWC(file):
    df = pd.read_csv(file)
    df['FLOPs'] = (2 * df['KH'] * df['KW'] + 1) * df['OH'] * df['OW'] * df['OC']
    df = df[df['FLOPs'] > 3]
    df = df[df['T'] > 2]
    df['KHKW'] = df['KH'] * df['KW']
    df['OHOW'] = df['OH'] * df['OW']
    df['OHOC'] = df['OH'] * df['OC']
    df['OWOC'] = df['OW'] * df['OC']
    df['OHOWOC'] = df['OH'] * df['OW'] * df['OC']
    df['KHKWOH'] = df['KHKW'] * df['OH']
    df['KHKWOW'] = df['KHKW'] * df['OW']
    df['KHKWOC'] = df['KHKW'] * df['OC']
    x_names = ['OH', 'OW', 'OC', 'KH', 'KW', 'KHKW', 'OHOW', 'OHOC', 'OWOC', 'OHOWOC', 'KHKWOH', 'KHKWOW', 'KHKWOC']
    target = 'T'
    linear_regression(df, x_names, target, file, frac = 0.60, random_state = 80)

def HWC_2_CHW(file):
    df = pd.read_csv(file)
    df = df[df['T'] > 2]
    df['TOTAL'] = df['IH'] * df['IW'] * df['IC']
    df['IHIW'] = df['IH'] * df['IW']
    df['IHIC'] = df['IH'] * df['IC']
    x_names = ['IH', 'IC', 'IHIW', 'IHIC', 'TOTAL']
    target = 'T'
    linear_regression(df, x_names, target, file, frac = 0.80, random_state = 100)

def CHW_2_HWC(file):
    df = pd.read_csv(file)
    df = df[df['T'] > 2]
    df['TOTAL'] = df['IH'] * df['IW'] * df['IC']
    df['IHIW'] = df['IH'] * df['IW']
    df['IHIC'] = df['IH'] * df['IC']
    x_names = ['IH', 'IC', 'IHIW', 'IHIC', 'TOTAL']
    target = 'T'
    linear_regression(df, x_names, target, file, frac = 0.80, random_state = 100)
    
def add(file):
    df = pd.read_csv(file)
    df = df[df['T'] > 2]
    x_names = ['size']
    target = 'T'
    linear_regression(df, x_names, target, file, frac = 0.80, random_state = 100)

 
def train(device_direct = 'collect_data_time/'):
    # convolution kernel size != 1 HWC
    # convolution_kernelN1_HWC('collect_data_time/convolution_kernelN1_HWC_15time.csv')
    # convolution kernel size 1x1 stride 1 pad 0 HWC
    # convolution_kernel1_stride1_pad0_HWC('collect_data_time/convolution_kernel1_stride1_pad0_HWC_15time.csv')
    # depthwise-convolution HWC
    
    data_map_fn = {
                    'convolution_kernelN1_HWC_15time.csv': convolution_kernelN1_HWC, #convolution kernel size != 1 HWC
                   'convolution_kernelN1_CHW_15time.csv': convolution_kernelN1_CHW,
                    'convolution_kernel1_stride1_pad0_HWC_15time.csv': convolution_kernel1_stride1_pad0_HWC, # convolution kernel size 1x1 stride 1 pad 0 HWC
                    'convolution_kernel1_stride1_pad0_CHW_15time.csv': convolution_kernel1_stride1_pad0_CHW,
                    'depthwise_HWC_15time.csv': depthwise_HWC_or_CHW, #CMSIS-NN
                   'depthwise_kernel3_stride1_CHW_15time.csv': depthwise_HWC_or_CHW, #MCUNET
                   'depthwise_kernel3_stride2_CHW_15time.csv': depthwise_HWC_or_CHW, #MCUNET
                #    'arm_avepool_q7_HWC_25time.csv': avepool_q7_HWC, # CMSiS-NN avepool_q7_HWC
                    'HWC_2_CHW_15time.csv': HWC_2_CHW,
                    'CHW_2_HWC_15time.csv': CHW_2_HWC,
                    'add_15time.csv': add,
                    'avgpooling_CHW_15time.csv': avgpooling_CHW_or_HWC, #MCUNET
                    'avgpooling_HWC_15time.csv': avgpooling_CHW_or_HWC, #MCUNET
                    'maxpooling_CHW_15time.csv': maxpooling_CHW_or_HWC, #MCUNET
                    'maxpooling_HWC_15time.csv': maxpooling_CHW_or_HWC, #MCUNET
                   }
    # depthwise-convolution
    device_direct = 'collect_data_time/m4/'
    for u,v in data_map_fn.items():
        file = device_direct + u
        if os.path.exists(file):
            v(file) 

if __name__ == '__main__':
    train()