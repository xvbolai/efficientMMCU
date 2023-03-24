from metric import PM10ACC
import pandas as pd



def convolution(name = 'SmallCifar'):
    
    df = pd.read_csv(f'./conv/convolution_kernel1_HWC_15time_{name}.csv')
    # df = df[df['OH'] == df['OW']]
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
    # x_names = ['M', 'K', 'N', 'MK', 'KN', 'MN', 'MKN']
    # target = 'T'
    df['predict'] = 0.015756537314687413*df['M'] -0.009027452543094219*df['K'] -0.010464224363182667*df['N'] +0.00024253853334692293*df['MK'] +0.00031400778352426597*df['KN'] +0.002533309148793464*df['MN'] +0.00032055671936570253*df['MKN'] -1.1844946743268565
    acc = PM10ACC(df['predict'].to_numpy(), df['T'].to_numpy())
    print(acc)
    
    number = len(df['predict'])
    lr_true_count = int(number * acc)
    # number = 0
    # lr_true_count = 0
    df = pd.read_csv(f'./conv/convolution_kernelN1_HWC_15time_{name}.csv')
    # df = df[df['OH'] == df['OW']]
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
    df['predict'] = -0.26589161450439713 *df['M'] + 0.0074355989302599184 *df['K'] -0.007354361926479533*df['N'] + 0.0009175239957340771*df['MK'] -8.604854030795066e-05*df['KN'] + 0.02762481679302134*df['MN'] + 0.0004090813068415448*df['MKN'] + 0.44843322805468233
    acc = PM10ACC(df['predict'].to_numpy(), df['T'].to_numpy())
    print(acc)
    lr_true_count = int(len(df['predict']) * acc) + lr_true_count
    number = number + len(df['predict'])
    print(lr_true_count / number * 100)
    print(lr_true_count, number)
    
convolution(name = 'MnasNet')